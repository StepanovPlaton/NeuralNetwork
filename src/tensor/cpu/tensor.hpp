#pragma once

#include "../tensor.hpp"

#include <vector>

template <typename T, int Dim> class Tensor : public ITensor<T, Dim> {
private:
  std::vector<T> data_;

public:
  typedef class ITensor<T, Dim> ITensor;

  using ITensor::axes_;
  using ITensor::checkAxisInDim;
  using ITensor::checkItHasSameShape;
  using ITensor::computeIndex;
  using ITensor::getSize;
  using ITensor::shape_;

  Tensor() = delete;
  Tensor(const std::array<size_t, Dim> &shape);
  Tensor(const std::array<size_t, Dim> &shape, T value);
  Tensor(const std::array<size_t, Dim> &shape, const std::vector<T> &data);
  Tensor(const std::array<size_t, Dim> &shape, T min, T max);

  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;
  ~Tensor() = default;

  T &operator[](size_t i);
  const T &operator[](size_t i) const;
  template <typename... Indices> T &operator()(Indices... indices);
  template <typename... Indices> const T &operator()(Indices... indices) const;

  using ITensor::operator+;
  using ITensor::operator-;

  Tensor operator+() const override;
  Tensor operator-() const override;

  Tensor &operator+=(const T scalar) override;

  Tensor &operator*=(const T scalar) override;

  Tensor &operator+=(const Tensor &other) override;

  Tensor &operator*=(const Tensor &other) override;

  Tensor<T, Dim == 1 ? 0 : 2> operator%(const Tensor &other) const;

  std::string toString() const override;
};

#include "tensor.tpp"

#include "../fabric.hpp"
