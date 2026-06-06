import numpy as np
from scipy.linalg import lu

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


p1 = Party(A, k=2, l=3)
print(p1.k)
print(p1.l)