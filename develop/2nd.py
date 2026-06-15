import numpy as np
from scipy.linalg import lu
import time

# numpy: math library for matrix operations, imported as 'np' for shorter usage
# lu: PLU decomposition function from scipy, so we don't have to calculate it manually


class Matrix:
    # Blueprint for matrix object, using OOP to keep code clean and easy to reuse

    def __init__(self, m, n):
        # runs automatically when Matrix is created
        # stores all parameters and reserves empty spot for matrix data
        self.m = m        # store number of rows
        self.n = n        # store number of columns
        self.data = None  # reserve empty spot for matrix numbers, filled later by generate()

    def generate(self):
        # randomly fill matrix with 0s and 1s (binary) and store in self.data
        self.data = np.random.randint(0, 2, size=(self.m, self.n))

    def is_full_rank(self):
        # use PLU decomposition to find rank of matrix
        # check diagonal of U for non-zero pivots
        # if number of non-zero pivots == n → matrix is full rank ✅
        P, L, U = lu(self.data)
        rank = np.sum(np.abs(np.diag(U)) > 1e-10)
        return rank == self.n

    def generate_full_rank(self):
        # keep generating random matrix until we get one that is full rank
        # only stops when is_full_rank() returns True
        while True:
            self.generate()
            if self.is_full_rank():
                break

    def transpose(self):
        # flip the matrix — rows become columns, columns become rows
        # e.g. shape 6x3 becomes 3x6
        return self.data.T


class Party:
    # Blueprint for each party (P1 or P2) in the key exchange

    def __init__(self, A, k, l):
        # runs automatically when Party is created
        # stores all the parameters and reserves empty spots for f, S, E, U, b
        self.A = A      # shared matrix between P1 and P2
        self.k = k      # number of non-zero elements
        self.l = l      # number of columns for S and E
        self.f = None   # secret vector, filled later by generate_f()
        self.S = None   # secret matrix, filled later by generate_S()
        self.E = None   # error matrix, filled later by generate_E()
        self.U = None   # public value, filled later by compute_U()
        self.b = None   # public value, filled later by compute_b()
        self.K = None   # secret key, filled later by compute_key()

    def generate_f(self):
        # create secret binary vector f, size m×1
        # k random positions set to 1, rest are 0
        f = np.zeros(self.A.m, dtype=int)                              # create empty vector size m
        positions = np.random.choice(self.A.m, self.k, replace=False)  # pick k random positions, no repeats
        f[positions] = 1                                                # set those positions to 1
        self.f = f.reshape(-1, 1)                                       # make it vertical (m×1)

    def generate_S(self):
        # create secret random binary matrix S, size n×l
        # elements are randomly 0 or 1
        self.S = np.random.randint(0, 2, size=(self.A.n, self.l))

    def generate_E(self):
        # create error matrix E, size m×l, start with all zeros
        # each column gets k random positions set to 1 (independently)
        E = np.zeros((self.A.m, self.l), dtype=int)  # start all zeros
        for col in range(self.l):                     # loop through each column
            positions = np.random.choice(self.A.m, self.k, replace=False)  # pick k random positions, no repeats
            E[positions, col] = 1                     # set those positions to 1
        self.E = E                                    # store in object

    def compute_U(self):
        # compute U = A·S + E mod 2
        # U is the public value sent to the other party
        self.U = (self.A.data @ self.S + self.E) % 2

    def compute_b(self):
        # compute b = Aᵀ·f mod 2
        # b is the public value sent to the other party
        self.b = (self.A.transpose() @ self.f) % 2

    def compute_key(self, other):
        # compute secret key K using other party's U and b
        # K = other.Uᵀ·f + Sᵀ·other.b mod 2
        # 'other' is the opposite party (P1 uses P2's values, P2 uses P1's values)
        self.K = (other.U.T @ self.f + self.S.T @ other.b) % 2

    def maj(self):
        # majority voting function — reconciliation step
        # count non-zero elements in K
        # if >= l/2 → return 1, else → return 0
        non_zero = np.count_nonzero(self.K)
        return 1 if non_zero >= self.l / 2 else 0


class LWEKeyExchange:
    # Main class that runs the full LWE key exchange system

    def __init__(self, m, n, k, l):
        # runs automatically when LWEKeyExchange object is created
        # stores all parameters and reserves empty spots for A, p1, p2
        self.m = m      # number of rows
        self.n = n      # number of columns
        self.k = k      # number of non-zero elements
        self.l = l      # number of columns for S and E
        self.A = None   # shared matrix, filled later by setup()
        self.p1 = None  # party 1, filled later by run()
        self.p2 = None  # party 2, filled later by run()

    def setup(self):
        # create shared matrix A and make sure it is full rank before use
        self.A = Matrix(self.m, self.n)       # create Matrix object
        self.A.generate_full_rank()           # keep generating until full rank

    def run(self):
        # create P1 and P2 with shared matrix A
        # each party generates their own secret f, S, E
        # each party computes their public U and b
        self.p1 = Party(self.A, self.k, self.l)
        self.p1.generate_f()
        self.p1.generate_S()
        self.p1.generate_E()
        self.p1.compute_U()
        self.p1.compute_b()

        self.p2 = Party(self.A, self.k, self.l)
        self.p2.generate_f()
        self.p2.generate_S()
        self.p2.generate_E()
        self.p2.compute_U()
        self.p2.compute_b()

        # each party computes key using the OTHER party's public values
        self.p1.compute_key(self.p2)
        self.p2.compute_key(self.p1)

    def experiment(self, num_experiments=10000, num_runs=10000):
        # run the key exchange num_experiments times
        # each experiment runs num_runs times and counts how many times maj(K1) == maj(K2)
        # find the average success rate across all experiments
        results = []

        for exp in range(num_experiments):
            count = 0
            self.setup()  # generate new A for each experiment

            for i in range(num_runs):
                self.run()
                if self.p1.maj() == self.p2.maj():
                    count += 1

            results.append(count)

            if (exp + 1) % 100 == 0:
                print(f"Experiment {exp+1}/{num_experiments} done...")

        average = np.mean(results) / num_runs
        print(f"\nAverage success rate: {average:.4f}")
        print(f"Average times maj(K1)==maj(K2): {np.mean(results):.1f}/{num_runs}")

        # save results to file
        with open("results.txt", "w") as f:

            f.write(f"Parameters: m={self.m}, n={self.n}, k={self.k}, l={self.l}\n")
            f.write(f"Experiments: {num_experiments}, Runs per experiment: {num_runs}\n")
            f.write(f"Average success rate: {average:.4f}\n")
            f.write(f"Average times maj(K1)==maj(K2): {np.mean(results):.1f}/{num_runs}\n")
            f.write(f"All results: {results}\n")
        print("Results saved to results.txt ✅")
        return average



# --- Run with professor's parameters ---
lwe = LWEKeyExchange(m=64, n=32, k=8, l=1000)

start = time.time()
lwe.experiment(num_experiments=1, num_runs=1000)  # small test first to estimate time
end = time.time()

elapsed = end - start
print(f"\nTime for 1 experiment x 1000 runs: {elapsed:.1f} seconds")
print(f"Estimated for 1000x1000: {elapsed * 100000 / 60:.0f} minutes")