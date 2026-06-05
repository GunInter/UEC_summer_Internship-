import numpy as np

class Matrix:

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.data = None 

    def generate(self):
        self.data = np.random.randint(0, 2, size=(self.m, self.n))

A = Matrix(6, 3)
A.generate()

print('Matrix columns:', A.m)
print('Matrix rows:', A.n)

print('Matrix data:',)
print(A.data)
