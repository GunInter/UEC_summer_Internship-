import numpy as np


# inputs of the matrix dimensions :)

n = int(input('Pls enter value n:'))
k = int(input('Pls enter value k:'))
m = int(input('Pls enter value m:'))

# create empty matrix
matrix = np.zeros((n, m), dtype=int)

# loop m times to fill each column
for i in range(m):
    
    positions = np.random.choice(n, k, replace=False)
    x = np.zeros(n, dtype=int)
    x[positions] = 1
    matrix[ :, i] = x  # save column i
    
    print('Positions:')
    print(positions)
    print('Column:')
    print(x)

print('Generated matrix:')
print(matrix)

