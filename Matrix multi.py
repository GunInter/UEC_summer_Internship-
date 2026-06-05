import numpy as np


# inputs of the matrix dimensions :)
m = int(input('Pls enter value m:'))
n = int(input('Pls enter value n:'))
k = int(input('Pls enter value k:'))

# now m and n exist, so we can use them
A = np.random.randint(0, 2, (m, n))
B = np.random.randint(0, 2, (n, k))

# check if the inner dimensions match for multiplication
if A.shape[1] != B.shape[0]:

    print("Error: cannot multiply, sizes don't match!")

else:

    print('Matrix A:')
    print(A)
    
    print('Matrix B:')
    print(B)

    C = (A @ B) % 2
    print('Matrix C:')
    print(C)