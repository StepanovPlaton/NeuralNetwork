#include <cstddef>

template <typename T, int Dim> class Tensor;

class Tensors {
  Tensors() = delete;

public:
  template <typename T, typename... Args> static auto empty(Args... args) {
    return Tensor<T, sizeof...(Args)>({static_cast<size_t>(args)...});
  }

  template <typename T, typename... Args> static auto zero(Args... args) {
    return Tensor<T, sizeof...(Args)>({static_cast<size_t>(args)...}, T(0));
  }

  template <typename T, typename... Args> static auto rand(Args... args) {
    return Tensor<T, sizeof...(Args)>({static_cast<size_t>(args)...}, T(0),
                                      T(1));
  }
};
