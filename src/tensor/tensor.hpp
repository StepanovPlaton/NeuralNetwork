#include <array>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T, int Dim> class Tensor {
private:
  std::array<size_t, Dim> shape_;
  std::array<int, Dim> axes_;
  std::vector<T> data_;

  template <typename... Indices> size_t computeIndex(Indices... indices) const {
    static_assert(sizeof...(Indices) == Dim, "Invalid number of indices");
    std::array<size_t, Dim> indicesArray = {static_cast<size_t>(indices)...};
    std::array<size_t, Dim> axesIndices;
    for (int i = 0; i < Dim; ++i)
      axesIndices[axes_[i]] = indicesArray[i];
    size_t index = 0;
    size_t stride = 1;
    for (int i = Dim - 1; i >= 0; --i) {
      index += axesIndices[i] * stride;
      stride *= shape_[i];
    }
    return index;
  }

  void checkItHasSameShape(const Tensor &other) {
    if (getShape() != other.getShape())
      throw std::invalid_argument("Tensor shapes must match");
  }
  void checkAxisInDim(int axis) {
    if (axis < 0 || axis >= Dim)
      throw std::invalid_argument("Invalid axis index");
  }

public:
  Tensor() = delete;
  Tensor(const std::array<size_t, Dim> &shape) {
    for (size_t d : shape)
      if (d == 0)
        throw std::invalid_argument("Invalid shape");
    shape_ = shape;
    for (int i = 0; i < Dim; ++i)
      axes_[i] = i;
    size_t total_size = 1;
    for (size_t dim : shape)
      total_size *= dim;
    data_.resize(total_size);
  }
  Tensor(const std::array<size_t, Dim> &shape, T fill) : Tensor(shape) {
    std::fill(data_.begin(), data_.end(), fill);
  }
  Tensor(const std::array<size_t, Dim> &shape, const std::vector<T> &data)
      : Tensor(shape) {
    if (data.size() != data_.size())
      throw std::invalid_argument("Invalid data size");
    data_ = data;
  }
  Tensor(const std::array<size_t, Dim> &shape, T min, T max) : Tensor(shape) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> dis(min, max);
      for (auto &element : data_)
        element = dis(gen);
    } else if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min, max);
      for (auto &element : data_)
        element = dis(gen);
    } else
      throw std::invalid_argument("Invalid randomized type");
  }

  Tensor(const Tensor &other)
      : shape_(other.shape_), axes_(other.axes_), data_(other.data_) {}
  Tensor &operator=(const Tensor &other) {
    shape_ = other.shape_;
    axes_ = other.axes_;
    data_ = other.data_;
    return *this;
  }
  Tensor(Tensor &&other) noexcept
      : shape_(std::move(other.shape_)), axes_(std::move(other.axes_)),
        data_(std::move(other.data_)) {}
  Tensor &operator=(Tensor &&other) noexcept {
    shape_ = std::move(other.shape_);
    axes_ = std::move(other.axes_);
    data_ = std::move(other.data_);
    return *this;
  }
  ~Tensor() = default;

  const std::array<int, Dim> &getAxes() const { return axes_; }
  const std::vector<T> &getData() const { return data_; }
  size_t getSize() const { return data_.size(); }
  const std::array<size_t, Dim> getShape() const {
    std::array<size_t, Dim> result;
    for (int i = 0; i < Dim; ++i)
      result[i] = shape_[axes_[i]];
    return result;
  }

  T &operator[](size_t i) { return data_[i]; }
  const T &operator[](size_t i) const { return data_[i]; }

  template <typename... Indices> T &operator()(Indices... indices) {
    return data_[computeIndex(indices...)];
  }
  template <typename... Indices> const T &operator()(Indices... indices) const {
    return data_[computeIndex(indices...)];
  }

  Tensor &transpose(const std::array<int, Dim> &new_axes) {
    std::array<bool, Dim> used{};
    for (int axis : new_axes) {
      checkAxisInDim(axis);
      if (used[axis])
        throw std::invalid_argument("Duplicate axis index");
      used[axis] = true;
    }
    axes_ = new_axes;
    return *this;
  }
  Tensor &transpose(int axis_a, int axis_b) {
    checkAxisInDim(axis_a);
    checkAxisInDim(axis_b);
    if (axis_a == axis_b)
      throw std::invalid_argument("Duplicate axis index");
    std::swap(axes_[axis_a], axes_[axis_b]);
    return *this;
  }
  Tensor &t() {
    static_assert(Dim >= 2, "Can't change the only axis");
    std::swap(axes_[Dim - 1], axes_[Dim - 2]);
    return *this;
  }

  Tensor operator+() const { return *this; }
  Tensor operator-() const {
    Tensor result = *this;
    for (T &e : result.data_)
      e = -e;
    return result;
  }

  Tensor &operator+=(const T &scalar) {
    for (T &e : data_)
      e += scalar;
    return *this;
  }
  Tensor operator+(const T &scalar) const {
    Tensor result = *this;
    result += scalar;
    return result;
  }
  friend Tensor operator+(const T &scalar, const Tensor &tensor) {
    return tensor + scalar;
  }

  Tensor &operator-=(const T &scalar) {
    for (T &e : data_)
      e -= scalar;
    return *this;
  }
  Tensor operator-(const T &scalar) const {
    Tensor result = *this;
    result -= scalar;
    return result;
  }
  friend Tensor operator-(const T &scalar, const Tensor &tensor) {
    Tensor result = tensor;
    for (T &e : result.data_)
      e = scalar - e;
    return result;
  }

  Tensor &operator*=(const T &scalar) {
    for (T &e : data_)
      e *= scalar;
    return *this;
  }
  Tensor operator*(const T &scalar) const {
    Tensor result = *this;
    result *= scalar;
    return result;
  }
  friend Tensor operator*(const T &scalar, const Tensor &tensor) {
    return tensor * scalar;
  }

  Tensor &operator/=(const T &scalar) {
    if (scalar == T(0))
      throw std::invalid_argument("Division by zero");
    for (T &e : data_)
      e /= scalar;
    return *this;
  }
  Tensor operator/(const T &scalar) const {
    Tensor result = *this;
    result /= scalar;
    return result;
  }

  Tensor &operator+=(const Tensor &other) {
    checkItHasSameShape(other);
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] += other.data_[i];
    return *this;
  }
  Tensor operator+(const Tensor &other) const {
    Tensor result = *this;
    result += other;
    return result;
  }

  Tensor &operator-=(const Tensor &other) {
    checkItHasSameShape(other);
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] -= other.data_[i];
    return *this;
  }
  Tensor operator-(const Tensor &other) const {
    Tensor result = *this;
    result -= other;
    return result;
  }

  Tensor &operator*=(const Tensor &other) {
    checkItHasSameShape(other);
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] *= other.data_[i];
    return *this;
  }
  Tensor operator*(const Tensor &other) const {
    Tensor result = *this;
    result *= other;
    return result;
  }

  Tensor<T, Dim == 1 ? 0 : 2> operator%(const Tensor &other) const {
    static_assert(Dim == 1 || Dim == 2,
                  "Inner product is only defined for vectors and matrices");
    if constexpr (Dim == 1) {
      if (data_.size() != other.data_.size())
        throw std::invalid_argument(
            "Vector sizes must match for inner product");
      T result_val = T(0);
      for (size_t i = 0; i < data_.size(); ++i)
        result_val += data_[i] * other.data_[i];
      return Tensor<T, 0>({}, {result_val});
    } else if constexpr (Dim == 2) {
      if (shape_[axes_[1]] != other.shape_[other.axes_[0]])
        throw std::invalid_argument(
            "Matrix dimensions must match for multiplication");
      size_t m = shape_[axes_[0]];
      size_t n = shape_[axes_[1]];
      size_t p = other.shape_[other.axes_[1]];
      Tensor<T, 2> result({m, p}, T(0));
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
          T sum = T(0);
          for (size_t k = 0; k < n; ++k)
            sum += (*this)(i, k) * other(k, j);
          result(i, j) = sum;
        }
      }
      return result;
    }
  }

  void print() const {
    if constexpr (Dim == 0) {
      std::cout << "Scalar<" << typeid(T).name() << ">: " << data_[0]
                << std::endl;
    } else if constexpr (Dim == 1) {
      std::cout << "Vector<" << typeid(T).name() << ">(" << shape_[0] << "): [";
      for (size_t i = 0; i < data_.size(); ++i) {
        std::cout << data_[i];
        if (i < data_.size() - 1)
          std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    } else if constexpr (Dim == 2) {
      std::cout << "Matrix<" << typeid(T).name() << ">(" << shape_[axes_[0]]
                << "x" << shape_[axes_[1]] << "):" << std::endl;
      for (size_t i = 0; i < shape_[axes_[0]]; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < shape_[axes_[1]]; ++j) {
          std::cout << (*this)(i, j);
          if (j < shape_[axes_[1]] - 1)
            std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }
    } else {
      std::cout << "Tensor" << Dim << "D<" << typeid(T).name() << ">" << "[";
      for (size_t i = 0; i < Dim; ++i) {
        std::cout << shape_[axes_[i]];
        if (i < Dim - 1)
          std::cout << "x";
      }
      std::cout << "]: [";
      size_t show = std::min(data_.size(), size_t(10));
      for (size_t i = 0; i < show; ++i) {
        std::cout << data_[i];
        if (i < show - 1)
          std::cout << ", ";
      }
      if (data_.size() > 10)
        std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
  }
};

template <typename T> using Scalar = Tensor<T, 0>;
template <typename T> using Vector = Tensor<T, 1>;
template <typename T> using Matrix = Tensor<T, 2>;

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