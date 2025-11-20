from tensor.tensor import *
import numpy as np
import time

if (MODE == PLATFORM.OPENCL):
    init("./tensor/")

a = Matrix([1024, 1024], 1)
b = Matrix([1024, 1024], 1)


def benchmark_tensor():
    c = ((a @ b) @ (a @ b)) @ ((a @ b) @ (a @ b))
    return c


a_np = np.ones([1024, 1024], dtype=np.float32)
b_np = np.ones([1024, 1024], dtype=np.float32)


def benchmark_numpy():
    c = ((a_np @ b_np) @ (a_np @ b_np)) @ ((a_np @ b_np) @ (a_np @ b_np))
    return c


# Многократное выполнение для более точного измерения
iterations = 5

print("Бенчмарк Tensor:")
tensor_times = []
for i in range(iterations):
    start = time.time()
    result_tensor = benchmark_tensor()
    print(result_tensor)
    tensor_times.append(time.time() - start)

print("Бенчмарк NumPy:")
numpy_times = []
for i in range(iterations):
    start = time.time()
    result_numpy = benchmark_numpy()
    print(result_numpy)
    numpy_times.append(time.time() - start)

print(
    f"\nСреднее время Tensor: {np.mean(tensor_times):.4f} ± {np.std(tensor_times):.4f} сек")
print(
    f"Среднее время NumPy: {np.mean(numpy_times):.4f} ± {np.std(numpy_times):.4f} сек")

ratio = np.mean(numpy_times) / np.mean(tensor_times)
if ratio > 1:
    print(f"Tensor быстрее в {ratio:.2f} раз")
else:
    print(f"NumPy быстрее в {1/ratio:.2f} раз")
