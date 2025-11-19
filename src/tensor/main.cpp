#ifdef USE_OPENCL
#include "opencl/tensor.hpp"
OpenCL openCL;
// TODO: GENERIC KERNELS
// TODO: Scalar mult
#elif USE_CPU
#include "cpu/tensor.hpp"
#endif

#include <iostream>

// TODO: TMult >2

int main() {
#ifdef USE_OPENCL
  openCL.init("./");
#endif

  Tensor<float, 2> a = Tensor<float, 2>({32, 32}, 2);
  std::cout << a.toString();
  return 0;
}
