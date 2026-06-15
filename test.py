import cupy as cp

# test GPU
a = cp.array([1, 2, 3])
print(a)
print("CuPy working on:", cp.cuda.Device())