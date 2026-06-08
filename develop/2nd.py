import numpy as np
from scipy.linalg import lu

class Matrix:

    def __init__(self, m, n):

        self.m = m
        self.n = n
        self.data = None 

    def generate(self):

        self.data = np.random.randint(0, 2, size=(self.m, self.n))


    def is_full_rank(self):

        P, L, U = lu(self.data)
        rank = np.sum(np.abs(np.diag(U)) > 1e-10)
        return rank == self.n
    
    def generate_full_rank(self):

        while True:
            self.generate()

            if self.is_full_rank():
                 break
            
    def transpose(self):
        return self.data.T




class Party:
    def __init__(self, A, k, l):

        self.A = A      # the shared matrix
        self.k = k      # number of non-zero elements
        self.l = l      # number of columns for S and E
        self.f = None
        self.S = None
        self.E = None
        self.U = None
        self.b = None

    def generate_S(self):
        self.S = np.random.randint(0, 2, size=(self.A.n, self.l))

    def generate_f(self):

        f = np.zeros(self.A.m, dtype=int)  # start with all zeros
        positions = np.random.choice(self.A.m, self.k, replace=False)  # pick k random positions
        f[positions] = 1                    # set those positions to 1
        self.f = f.reshape(-1, 1)

    def generate_E(self):

        E = np.zeros((self.A.m, self.l), dtype=int)  # start all zeros

        for col in range(self.l):    # for each column

            positions = np.random.choice(self.A.m, self.k, replace=False)  # pick k random positions
            E[positions, col] = 1                     # set those positions to 1
        self.E = E
    
    def compute_U(self):

        self.U = (self.A.data @ self.S + self.E) % 2

    def compute_b(self):

        self.b = (self.A.transpose() @ self.f) % 2

    def compute_key(self, other):

        self.K = (other.U.T @ self.f + self.S.T @ other.b) % 2



class LWEKeyExchange:

    def __init__(self, m, n, k, l):
        self.m = m
        self.n = n
        self.k = k
        self.l = l
        self.A = None
        self.p1 = None
        self.p2 = None

    def setup(self):

        self.A = Matrix(self.m, self.n)
        self.A.generate_full_rank()
        print("A generated! Full rank:", self.A.is_full_rank())

    def run(self):

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

        self.p1.compute_key(self.p2)
        self.p2.compute_key(self.p1)    

    def verify(self):

        if np.array_equal(self.p1.K, self.p2.K):
            print("✅ K1 == K2! Key exchange successful!")

        else:
            print("❌ K1 != K2! Key exchange failed!")
        print("K1 =\n", self.p1.K)
        print("K2 =\n", self.p2.K)


lwe = LWEKeyExchange(m=6, n=3, k=2, l=3)
lwe.setup()
lwe.run()
lwe.verify()



