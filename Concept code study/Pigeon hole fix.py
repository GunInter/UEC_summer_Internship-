import numpy as np

n = int(input('Pls enter value n:'))
k = int(input('Pls enter value k:'))
m = int(input('Pls enter value m:'))

# create entire matrix in one go!
matrix = np.zeros((n, m), dtype=int)

for i in range(m):
    matrix[np.random.choice(n, k, replace=False), i] = 1

print('Generated matrix:')
print(matrix)