#pragma once

#include <array>
#include <cstddef>
#include <string>

template <typename T, int Dim> class Tensor;

template <typename T, int Dim> class ITensor {
protected:
  std::array<size_t, Dim> shape_;
  std::array<int, Dim> axes_;

  template <typename... Indices> size_t computeIndex(Indices... indices) const;

  void checkItHasSameShape(const ITensor &other) const;
  void checkAxisInDim(int axis) const;

public:
  typedef class Tensor<T, Dim> Tensor;

  ITensor() = delete;
  ITensor(const std::array<size_t, Dim> &shape);
  ITensor(const ITensor &other);
  ITensor &operator=(const ITensor &other);
  ITensor(ITensor &&other) noexcept;
  ITensor &operator=(ITensor &&other) noexcept;
  ~ITensor() = default;

  const std::array<int, Dim> &getAxes() const;
  const std::array<size_t, Dim> getShape() const;
  size_t getSize() const;

  Tensor &transpose(const std::array<int, Dim> &new_axes);
  Tensor &transpose(int axis_a, int axis_b);
  Tensor &t();

  // === Operators ===
  virtual Tensor operator+() const = 0;
  virtual Tensor operator-() const = 0;

  virtual Tensor &operator+=(const T &scalar) = 0;
  virtual Tensor &operator*=(const T &scalar) = 0;

  virtual Tensor &operator+=(const Tensor &other) = 0;
  virtual Tensor &operator*=(const Tensor &other) = 0;

  Tensor operator+(const T &scalar) const;
  friend Tensor operator+(const T &scalar, const Tensor &tensor) {
    return tensor + scalar;
  }

  Tensor &operator-=(const T &scalar);
  Tensor operator-(const T &scalar) const;
  friend Tensor operator-(const T &scalar, const Tensor &tensor) {
    return tensor + (-scalar);
  }

  Tensor operator*(const T &scalar) const;
  friend Tensor operator*(const T &scalar, const Tensor &tensor) {
    return tensor * scalar;
  }

  Tensor &operator/=(const T &scalar);
  Tensor operator/(const T &scalar) const;

  Tensor operator+(const Tensor &other) const;

  Tensor &operator-=(const Tensor &other);
  Tensor operator-(const Tensor &other) const;

  Tensor operator*(const Tensor &other) const;

  // === Utils ===
  virtual std::string toString() const = 0;
};

#include "tensor.tpp"
