// #include "cpu/tensor.hpp"
#include "opencl/tensor.hpp"

#include <iostream>

// TODO: GENERIC KERNELS
// TODO: Scalar mult

int main() {
  Tensor<float, 2> a = Tensor<float, 2>({2, 4});
  std::cout << a.toString();
  return 0;
}
