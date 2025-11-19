from tensor.tensor import *

if (MODE == PLATFORM.OPENCL):
    init("./tensor/")

a = Matrix([1024, 1024], 1)
a += 1
b = Matrix([1024, 1024], 1)
c = a @ b
print(c)
