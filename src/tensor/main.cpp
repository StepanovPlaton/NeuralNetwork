#ifdef USE_OPENCL
#include "opencl/tensor.hpp"
OpenCL openCL;
// TODO: GENERIC KERNELS
// TODO: Scalar mult
#elif USE_CPU
#include "cpu/tensor.hpp"
#endif

#include <chrono>
#include <functional>
#include <iostream>

// TODO: TMult >2

class Profiler {
public:
  static void measure(const std::string &operation, std::function<void()> op) {
    auto start = std::chrono::high_resolution_clock::now();
    op();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << operation << ": " << duration.count() << " μs\n";
  }
};

int main() {
#ifdef USE_OPENCL
  openCL.init("./");
#endif

  Tensor<float, 2> a = Tensor<float, 2>({4096 * 2, 4096 * 2}, 1);
  Tensor<float, 2> b = Tensor<float, 2>({4096 * 2, 4096 * 2}, 1);

  Profiler::measure("Matrix multiplication", [&]() { auto result = a % b; });
  std::cout << a.toString();
  return 0;
}
