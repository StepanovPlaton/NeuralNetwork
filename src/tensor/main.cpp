// #include "cpu/tensor.hpp"
#include "opencl/tensor.hpp"

#include <iostream>

// TODO: GENERIC KERNELS
// TODO: Scalar mult
// TODO: TMult >2

OpenCL openCL;

int main() {
  Tensor<float, 2> a = Tensor<float, 2>({8192, 8192}, 1);
  Tensor<float, 2> b = Tensor<float, 2>({8192, 8192}, 1);
  auto c = a % b;
  Tensor<float, 2> d = Tensor<float, 2>(c);
  d += 1;
  std::cout << d.toString();
  return 0;
}
