import torch
import numpy as np
import time

# torch: PyTorch library for GPU matrix operations, works with RTX 5060
# numpy: used for averaging results at the end
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class Matrix:
    # Blueprint for matrix object

    def __init__(self, m, n):
        self.m = m        # number of rows
        self.n = n        # number of columns
        self.data = None  # filled later by generate()

    def generate(self):
        # float32 required by matrix_rank
        self.data = torch.randint(0, 2, size=(self.m, self.n), device=device, dtype=torch.float32)

    def is_full_rank(self):
        # check rank on GPU — much faster than scipy on CPU
        rank = torch.linalg.matrix_rank(self.data)
        return rank.item() == self.n

    def generate_full_rank(self):
        while True:
            self.generate()
            if self.is_full_rank():
                break


class LWEKeyExchange:
    # Main class that runs the full LWE key exchange system

    def __init__(self, m, n, k, l):
        self.m = m      # number of rows
        self.n = n      # number of columns
        self.k = k      # number of non-zero elements in f and E
        self.l = l      # number of columns for S and E
        self.A = None

    def setup(self):
        # create shared matrix A
        # skip full rank check — a random 512x256 binary matrix is virtually always full rank
        # checking rank 10000 times with SVD is slow and unnecessary
        self.A = Matrix(self.m, self.n)
        self.A.generate()

    def _gen_f_batch(self, bs):
        # generate bs secret f vectors — shape (bs, m, 1), float16
        # each f has exactly k ones in random positions
        f = torch.zeros(bs, self.m, 1, dtype=torch.float16, device=device)
        rand = torch.rand(bs, self.m, device=device)
        _, top_k = torch.topk(rand, self.k, dim=1)              # (bs, k)
        b_idx = torch.arange(bs, device=device).unsqueeze(1).expand(bs, self.k)
        f[b_idx.reshape(-1), top_k.reshape(-1), 0] = 1.0
        return f  # (bs, m, 1)

    def _gen_S_batch(self, bs):
        # generate bs secret S matrices — shape (bs, n, l), float16
        return torch.randint(0, 2, size=(bs, self.n, self.l), dtype=torch.float16, device=device)

    def _gen_E_positions(self, bs, chunk=50):
        # instead of dense E (bs, m, l), just store POSITIONS of the 1s — shape (bs, l, k)
        # process in chunks of columns to avoid one huge (bs, l, m) tensor in memory
        # chunk=50 means (bs, 50, m) at a time instead of (bs, 1000, m) — 20x less memory!
        parts = []
        for c in range(0, self.l, chunk):
            nc = min(chunk, self.l - c)
            rand = torch.rand(bs, nc, self.m, device=device)   # small chunk: (bs, 50, m)
            _, pos = torch.topk(rand, self.k, dim=2)           # (bs, 50, k)
            parts.append(pos)
        return torch.cat(parts, dim=1)  # (bs, l, k)

    def _E_transpose_f(self, positions, f_flat):
        # compute E^T @ f WITHOUT materializing dense E matrix
        # positions: (bs, l, k) — where E has 1s
        # f_flat: (bs, m) — f vector flattened
        # result: (bs, l, 1) — each element is sum of f at E's nonzero rows
        bs = positions.shape[0]
        b_idx = torch.arange(bs, device=device)[:, None, None].expand_as(positions)
        gathered = f_flat[b_idx, positions]          # (bs, l, k) — f values at E positions
        return (gathered.sum(dim=2, keepdim=True) % 2).half()  # (bs, l, 1)

    def run_batch(self, bs):
        # run bs key exchanges IN PARALLEL on GPU
        #
        # KEY MATH TRICK — expand K formula to avoid storing huge U matrices:
        #   K1 = U2^T*f1 + S1^T*b2
        #      = (AS2+E2)^T*f1 + S1^T*b2
        #      = S2^T*b1 + E2^T*f1 + S1^T*b2   (no U needed!)
        #   K2 = S1^T*b2 + E1^T*f2 + S2^T*b1
        #   shared = S1^T*b2 + S2^T*b1  (appears in both K1 and K2)
        #   K1 = shared + E2^T*f1
        #   K2 = shared + E1^T*f2
        #
        # This removes the huge U matrices (saves ~2GB VRAM per batch!)

        A  = self.A.data.half()   # (m, n) float16
        AT = A.T                  # (n, m) float16

        # generate secrets for both parties
        f1 = self._gen_f_batch(bs)        # (bs, m, 1)
        f2 = self._gen_f_batch(bs)
        S1 = self._gen_S_batch(bs)        # (bs, n, l)
        S2 = self._gen_S_batch(bs)
        pos1 = self._gen_E_positions(bs)  # (bs, l, k) sparse E1
        pos2 = self._gen_E_positions(bs)  # (bs, l, k) sparse E2

        # b = A^T * f mod 2  (small: bs, n, 1)
        b1 = torch.matmul(AT, f1) % 2    # (bs, n, 1)
        b2 = torch.matmul(AT, f2) % 2    # (bs, n, 1)

        # shared part: S1^T*b2 + S2^T*b1 mod 2  (small: bs, l, 1)
        term_A = torch.bmm(S1.transpose(1, 2), b2) % 2   # S1^T*b2
        term_B = torch.bmm(S2.transpose(1, 2), b1) % 2   # S2^T*b1
        shared = (term_A + term_B) % 2                    # (bs, l, 1)

        # error terms: E^T*f using sparse positions (no dense E matrix!)
        f1_flat = f1[:, :, 0]   # (bs, m)
        f2_flat = f2[:, :, 0]
        E2_f1 = self._E_transpose_f(pos2, f1_flat)   # (bs, l, 1)
        E1_f2 = self._E_transpose_f(pos1, f2_flat)   # (bs, l, 1)

        # final keys (both tiny: bs, l, 1)
        K1 = (shared + E2_f1) % 2
        K2 = (shared + E1_f2) % 2

        # ── Task 1: majority over 1024-bit vector → 1 bit per run ────────────
        # K_tilde = 1 if #{i: K_i = 1} >= l/2, else 0
        nz1 = (K1.reshape(bs, -1) != 0).sum(dim=1)
        nz2 = (K2.reshape(bs, -1) != 0).sum(dim=1)
        maj1 = (nz1 >= self.l / 2)   # (bs,) bool — Task 1 bit for party 1
        maj2 = (nz2 >= self.l / 2)   # (bs,) bool — Task 1 bit for party 2

        # return both tensors so experiment() can accumulate for Task 2
        return maj1, maj2

    def experiment(self, num_experiments=10000, num_runs=10000, batch_size=1000):
        task1_rates   = []   # per-experiment Task 1 agreement rate (~53.8%)
        task2_matches = []   # per-experiment Task 2 match (True/False, ~98%)

        for exp in range(num_experiments):
            self.setup()   # new A for each experiment

            all_maj1 = []  # collect all Task 1 bits across runs
            all_maj2 = []

            remaining = num_runs
            with torch.no_grad():   # no gradients needed — saves memory and time
                while remaining > 0:
                    bs = min(batch_size, remaining)
                    maj1, maj2 = self.run_batch(bs)
                    # move to CPU and store as bool list
                    all_maj1.append(maj1.cpu())
                    all_maj2.append(maj2.cpu())
                    remaining -= bs

            # stack into tensors of shape (num_runs,)
            all_maj1 = torch.cat(all_maj1)   # (num_runs,) bool
            all_maj2 = torch.cat(all_maj2)

            # ── Task 1: fraction of runs where maj(K1) == maj(K2) ────────────
            task1_agree = (all_maj1 == all_maj2).float().mean().item()
            task1_rates.append(task1_agree)

            # ── Task 2: majority of majority → single final key bit ───────────
            # K_hat = 1 if majority of {maj(K)_1 ... maj(K)_N} is 1, else 0
            K_hat1 = 1 if all_maj1.float().sum() >= num_runs / 2 else 0
            K_hat2 = 1 if all_maj2.float().sum() >= num_runs / 2 else 0
            task2_matches.append(K_hat1 == K_hat2)

            if (exp + 1) % 100 == 0:
                t1_avg = np.mean(task1_rates)
                t2_avg = np.mean(task2_matches)
                print(f"Exp {exp+1:5d}/{num_experiments} | "
                      f"Task1 rate: {t1_avg:.4f} (~53.8% expected) | "
                      f"Task2 match: {t2_avg:.4f} (~98% expected)")

        task1_final = np.mean(task1_rates)
        task2_final = np.mean(task2_matches)

        print(f"\n{'='*55}")
        print(f"Parameters: m={self.m}, n={self.n}, k={self.k}, l={self.l}")
        print(f"Experiments: {num_experiments}, Runs per experiment: {num_runs}")
        print(f"{'='*55}")
        print(f"Task 1 — single majority    : {task1_final:.4f}  (~0.538 expected)")
        print(f"Task 2 — majority of majority: {task2_final:.4f}  (~0.98  expected)")
        print(f"{'='*55}")

        with open("results.txt", "w") as f:
            f.write(f"Parameters: m={self.m}, n={self.n}, k={self.k}, l={self.l}\n")
            f.write(f"Experiments: {num_experiments}, Runs per experiment: {num_runs}\n\n")
            f.write(f"Task 1 — single majority     : {task1_final:.4f}  (~0.538 expected)\n")
            f.write(f"Task 2 — majority of majority: {task2_final:.4f}  (~0.98  expected)\n\n")
            f.write(f"Per-experiment Task1 rates: {task1_rates}\n")
            f.write(f"Per-experiment Task2 match: {task2_matches}\n")

        print("Results saved to results.txt")
        return task1_final, task2_final


# ── Run with professor's parameters ──────────────────────────────────────────
lwe = LWEKeyExchange(m=512, n=256, k=16, l=1024)

start = time.time()
lwe.experiment(num_experiments=10000, num_runs=10000, batch_size=1000)
end = time.time()

print(f"\nTotal time: {(end - start) / 60:.1f} minutes")
