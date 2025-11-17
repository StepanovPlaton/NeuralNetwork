#pragma once

#include "tensor.hpp"

#include <random>
#include <sstream>

// ===== CONSTRUCTORS =====
template <typename T, int Dim>
Tensor<T, Dim>::Tensor(const std::array<size_t, Dim> &shape) : ITensor(shape) {
  data_.resize(getSize());
}
template <typename T, int Dim>
Tensor<T, Dim>::Tensor(const std::array<size_t, Dim> &shape, T value)
    : Tensor(shape) {
  std::fill(data_.begin(), data_.end(), value);
}
template <typename T, int Dim>
Tensor<T, Dim>::Tensor(const std::array<size_t, Dim> &shape,
                       const std::vector<T> &data)
    : Tensor(shape) {
  if (data.size() != data_.size())
    throw std::invalid_argument("Invalid fill data size");
  data_ = data;
}
template <typename T, int Dim>
Tensor<T, Dim>::Tensor(const std::array<size_t, Dim> &shape, T min, T max)
    : Tensor(shape) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dis(min, max);
    for (T &e : data_)
      e = dis(gen);
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dis(min, max);
    for (T &e : data_)
      e = dis(gen);
  } else
    throw std::invalid_argument("Invalid randomized type");
}

template <typename T, int Dim>
Tensor<T, Dim>::Tensor(const Tensor &other)
    : ITensor(other), data_(other.data_) {}
template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator=(const Tensor &other) {
  ITensor::operator=(other);
  data_ = other.data_;
  return *this;
}
template <typename T, int Dim>
Tensor<T, Dim>::Tensor(Tensor &&other) noexcept
    : ITensor(std::move(other)), data_(std::move(other.data_)) {}
template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator=(Tensor &&other) noexcept {
  ITensor::operator=(std::move(other));
  data_ = std::move(other.data_);
  return *this;
}

// ===== GET/SET =====
template <typename T, int Dim> T &Tensor<T, Dim>::operator[](size_t i) {
  return data_[i];
}
template <typename T, int Dim>
const T &Tensor<T, Dim>::operator[](size_t i) const {
  return data_[i];
}
template <typename T, int Dim>
template <typename... Indices>
T &Tensor<T, Dim>::operator()(Indices... indices) {
  return data_[computeIndex(indices...)];
}
template <typename T, int Dim>
template <typename... Indices>
const T &Tensor<T, Dim>::operator()(Indices... indices) const {
  return data_[computeIndex(indices...)];
}

// ===== OPERATORS =====
template <typename T, int Dim>
Tensor<T, Dim> Tensor<T, Dim>::operator+() const {
  Tensor result = *this;
  for (T &e : result.data_)
    e = +e;
  return result;
}
template <typename T, int Dim>
Tensor<T, Dim> Tensor<T, Dim>::operator-() const {
  Tensor result = *this;
  for (T &e : result.data_)
    e = -e;
  return result;
}

template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator+=(const T &scalar) {
  for (T &e : data_)
    e += scalar;
  return *this;
}

template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator*=(const T &scalar) {
  for (T &e : data_)
    e *= scalar;
  return *this;
}

template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator+=(const Tensor &other) {
  checkItHasSameShape(other);
  for (size_t i = 0; i < data_.size(); ++i)
    data_[i] += other.data_[i];
  return *this;
}

template <typename T, int Dim>
Tensor<T, Dim> &Tensor<T, Dim>::operator*=(const Tensor &other) {
  checkItHasSameShape(other);
  for (size_t i = 0; i < data_.size(); ++i)
    data_[i] *= other.data_[i];
  return *this;
}

template <typename T, int Dim>
Tensor<T, Dim == 1 ? 0 : 2>
Tensor<T, Dim>::operator%(const Tensor &other) const {
  static_assert(Dim == 1 || Dim == 2,
                "Inner product is only defined for vectors and matrices");
  if constexpr (Dim == 1) {
    if (data_.size() != other.data_.size())
      throw std::invalid_argument("Vector sizes must match for inner product");
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

// ===== UTILS =====
template <typename T, int Dim> std::string Tensor<T, Dim>::toString() const {
  std::ostringstream oss;
  if constexpr (Dim == 0) {
    oss << "Scalar<" << typeid(T).name() << ">: " << data_[0];
  } else if constexpr (Dim == 1) {
    oss << "Vector<" << typeid(T).name() << ">(" << shape_[0] << "): [";
    for (size_t i = 0; i < data_.size(); ++i) {
      oss << data_[i];
      if (i < data_.size() - 1)
        oss << ", ";
    }
    oss << "]";
  } else if constexpr (Dim == 2) {
    oss << "Matrix<" << typeid(T).name() << ">(" << shape_[axes_[0]] << "x"
        << shape_[axes_[1]] << "):";
    for (size_t i = 0; i < shape_[axes_[0]]; ++i) {
      oss << "\n  [";
      for (size_t j = 0; j < shape_[axes_[1]]; ++j) {
        oss << (*this)(i, j);
        if (j < shape_[axes_[1]] - 1)
          oss << ", ";
      }
      oss << "]";
    }
  } else {
    oss << "Tensor" << Dim << "D<" << typeid(T).name() << ">" << "[";
    for (size_t i = 0; i < Dim; ++i) {
      oss << shape_[axes_[i]];
      if (i < Dim - 1)
        oss << "x";
    }
    oss << "]: [";
    size_t show = std::min(data_.size(), size_t(10));
    for (size_t i = 0; i < show; ++i) {
      oss << data_[i];
      if (i < show - 1)
        oss << ", ";
    }
    if (data_.size() > 10)
      oss << ", ...";
    oss << "]";
  }
  return oss.str();
}
