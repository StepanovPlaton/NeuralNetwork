#include "tensor.hpp"
#include <iostream>

int main() {
  Tensor<float, 2> a = Tensors::rand<float>(1, 3);
  std::cout << a.toString();
  return 0;
}
