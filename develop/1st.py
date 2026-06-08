# numpy: math library for matrix operations, imported as 'np' for shorter usage
# lu: PLU decomposition function from scipy, so we don't have to calculate it manually
import numpy as np
from scipy.linalg import lu

class Matrix:

    # Blueprint for matrix object, using OOP to keep code clean and easy to reuse
    def __init__(self, m, n):
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



A = Matrix(6, 3)
A.generate()
A.generate_full_rank()

print('Matrix columns:', A.m)
print('Matrix rows:', A.n)

print('Matrix data:',)
print(A.data)

print('Is the matrix full rank?', A.is_full_rank())
print("Transpose:\n", A.transpose())



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
    
    # def generate_E(self):

    #     self.E = np.zeros((self.A.m, self.l), dtype=int)  # all zeros!

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


p1 = Party(A, k=2, l=3)
print(p1.k)
print(p1.l)

p1.generate_f()
print("f =", p1.f)

p1.generate_S()
print("S =\n", p1.S)

p1.generate_E()
print("E =\n", p1.E)

p1.compute_U()
print("U =\n", p1.U)

p1.compute_b()
print("b =\n", p1.b)

p1 = Party(A, k=2, l=3)
p1.generate_f()
p1.generate_S()
p1.generate_E()
p1.compute_U()
p1.compute_b()

p2 = Party(A, k=2, l=3)
p2.generate_f()
p2.generate_S()
p2.generate_E()
p2.compute_U()
p2.compute_b()

p1.compute_key(p2)
p2.compute_key(p1)

print("K1 =\n", p1.K)
print("K2 =\n", p2.K)
print("K1 == K2?", np.array_equal(p1.K, p2.K))

count = 0
for i in range(100):
    p1 = Party(A, k=2, l=3)
    p1.generate_f()
    p1.generate_S()
    p1.generate_E()
    p1.compute_U()
    p1.compute_b()

    p2 = Party(A, k=2, l=3)
    p2.generate_f()
    p2.generate_S()
    p2.generate_E()
    p2.compute_U()
    p2.compute_b()

    p1.compute_key(p2)
    p2.compute_key(p1)

    if np.array_equal(p1.K, p2.K):
        count += 1

print(f"K1 == K2: {count}/100 times")

lwe = LWEKeyExchange(m=6, n=3, k=2, l=3)
print(lwe.m, lwe.n, lwe.k, lwe.l)

lwe = LWEKeyExchange(m=6, n=3, k=2, l=3)
lwe.setup()
lwe.run()
lwe.verify()

print("K1 =\n", lwe.p1.K)
print("K2 =\n", lwe.p2.K)



