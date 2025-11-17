#include "cpu/tensor.hpp"

#include <iostream>

int main() {
  Tensor<float, 2> a = Tensor<float, 2>({2, 4});
  std::cout << a.toString();
  return 0;
}
