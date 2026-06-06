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

    def generate_f(self):

        f = np.zeros(self.A.m, dtype=int)  # start with all zeros
        positions = np.random.choice(self.A.m, self.k, replace=False)  # pick k random positions
        f[positions] = 1                    # set those positions to 1
        self.f = f.reshape(-1, 1)

    def generate_S(self):
        self.S = np.random.randint(0, 2, size=(self.A.n, self.l))

p1 = Party(A, k=2, l=3)
print(p1.k)
print(p1.l)

p1.generate_f()
print("f =", p1.f)

p1.generate_S()
print("S =\n", p1.S)

