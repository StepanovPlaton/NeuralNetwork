#pragma once

#include "tensor.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

// ===== UTILS =====
template <typename T, int Dim>
template <typename... Indices>
size_t ITensor<T, Dim>::computeIndex(Indices... indices) const {
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

template <typename T, int Dim>
void ITensor<T, Dim>::checkItHasSameShape(const ITensor<T, Dim> &other) const {
  if (getShape() != other.getShape())
    throw std::invalid_argument("Tensor shapes must match");
}

template <typename T, int Dim>
void ITensor<T, Dim>::checkAxisInDim(int axis) const {
  if (axis < 0 || axis >= Dim)
    throw std::invalid_argument("Invalid axis index");
}

template <typename T, int Dim>
std::string ITensor<T, Dim>::format(std::vector<T> data) const {
  std::ostringstream oss;
  static auto formatValue = [](T value) -> std::string {
    std::ostringstream value_oss;
    if constexpr (std::is_floating_point_v<T>) {
      value_oss << std::fixed << std::setprecision(3) << value;
      std::string str = value_oss.str();
      if (str.find('.') != std::string::npos) {
        str = str.substr(0, str.find_last_not_of('0') + 1);
        if (str.back() == '.')
          str.pop_back();
      }
      return str;
    } else {
      value_oss << value;
      return value_oss.str();
    }
  };

  if constexpr (Dim == 0) {
    oss << "Scalar<" << typeid(T).name() << ">: " << formatValue(data[0]);
  } else if constexpr (Dim == 1) {
    oss << "Vector<" << typeid(T).name() << ">(" << shape_[0] << "): [";
    for (size_t i = 0; i < getSize(); ++i) {
      oss << formatValue(data[i]);
      if (i < getSize() - 1)
        oss << ", ";
    }
    oss << "]";
  } else if constexpr (Dim == 2) {
    const size_t rows = shape_[axes_[0]];
    const size_t cols = shape_[axes_[1]];
    oss << "Matrix<" << typeid(T).name() << ">(" << rows << "x" << cols << "):";

    const size_t MAX_FULL_ROWS = 8;
    const size_t MAX_FULL_COLS = 8;
    const size_t SHOW_FIRST = 3;
    const size_t SHOW_LAST = 3;
    bool show_abbreviated_rows = rows > MAX_FULL_ROWS;
    bool show_abbreviated_cols = cols > MAX_FULL_COLS;
    std::vector<std::string> formatted_values;
    size_t max_width = 0;
    for (size_t i = 0; i < rows; ++i) {
      if (show_abbreviated_rows && i >= SHOW_FIRST && i < rows - SHOW_LAST)
        continue;
      for (size_t j = 0; j < cols; ++j) {
        if (show_abbreviated_cols && j >= SHOW_FIRST && j < cols - SHOW_LAST)
          continue;
        std::string formatted = formatValue(data[i * cols + j]);
        formatted_values.push_back(formatted);
        max_width = std::max(max_width, formatted.length());
      }
    }
    for (size_t i = 0; i < rows; ++i) {
      if (show_abbreviated_rows && i >= SHOW_FIRST && i < rows - SHOW_LAST) {
        if (i == SHOW_FIRST) {
          oss << "\n  ";
          for (size_t j = 0; j < cols; ++j) {
            if (show_abbreviated_cols && j >= SHOW_FIRST &&
                j < cols - SHOW_LAST) {
              if (j == SHOW_FIRST)
                oss << std::string(max_width, '.') << ", ";
              continue;
            }
            oss << std::string(max_width, '.');
            if (!((!show_abbreviated_cols && j == cols - 1) ||
                  (show_abbreviated_cols &&
                   ((j < SHOW_FIRST && j == SHOW_FIRST - 1) || j == cols - 1))))
              oss << ", ";
          }
        }
        continue;
      }
      oss << "\n  [";
      for (size_t j = 0; j < cols; ++j) {
        if (show_abbreviated_cols && j >= SHOW_FIRST && j < cols - SHOW_LAST) {
          if (j == SHOW_FIRST)
            oss << " " << std::setw(max_width) << std::left << "..." << ", ";
          continue;
        }
        std::string formatted = formatValue(data[i * cols + j]);
        oss << std::setw(max_width) << std::left << formatted;
        if (!((!show_abbreviated_cols && j == cols - 1) ||
              (show_abbreviated_cols &&
               ((j < SHOW_FIRST && j == SHOW_FIRST - 1) || j == cols - 1))))
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
    size_t show = std::min(getSize(), size_t(10));
    for (size_t i = 0; i < show; ++i) {
      oss << formatValue(data[i]);
      if (i < show - 1)
        oss << ", ";
    }
    if (getSize() > 10)
      oss << ", ...";
    oss << "]";
  }
  return oss.str();
}

// ====== CONSTRUCT =====
template <typename T, int Dim>
ITensor<T, Dim>::ITensor(const std::array<size_t, Dim> &shape) {
  for (size_t d : shape)
    if (d == 0)
      throw std::invalid_argument("Invalid shape");
  shape_ = shape;
  for (int i = 0; i < Dim; ++i)
    axes_[i] = i;
}

template <typename T, int Dim>
ITensor<T, Dim>::ITensor(const ITensor &other)
    : shape_(other.shape_), axes_(other.axes_) {}

template <typename T, int Dim>
ITensor<T, Dim> &ITensor<T, Dim>::operator=(const ITensor &other) {
  shape_ = other.shape_;
  axes_ = other.axes_;
  return *this;
}
template <typename T, int Dim>
ITensor<T, Dim>::ITensor(ITensor &&other) noexcept
    : shape_(std::move(other.shape_)), axes_(std::move(other.axes_)) {}
template <typename T, int Dim>
ITensor<T, Dim> &ITensor<T, Dim>::operator=(ITensor &&other) noexcept {
  shape_ = std::move(other.shape_);
  axes_ = std::move(other.axes_);
  return *this;
}

// ===== GET/SET =====
template <typename T, int Dim>
const std::array<int, Dim> &ITensor<T, Dim>::getAxes() const {
  return axes_;
}
template <typename T, int Dim>
const std::array<size_t, Dim> ITensor<T, Dim>::getShape() const {
  std::array<size_t, Dim> result;
  for (int i = 0; i < Dim; ++i)
    result[i] = shape_[axes_[i]];
  return result;
}
template <typename T, int Dim> size_t ITensor<T, Dim>::getSize() const {
  size_t size = 1;
  for (size_t i = 0; i < shape_.size(); ++i)
    size *= shape_[i];
  return size;
};

// ===== TRANSPOSE =====
template <typename T, int Dim>
ITensor<T, Dim>::Tensor &
ITensor<T, Dim>::transpose(const std::array<int, Dim> &new_axes) {
  std::array<bool, Dim> used{};
  for (int axis : new_axes) {
    checkAxisInDim(axis);
    if (used[axis])
      throw std::invalid_argument("Duplicate axis index");
    used[axis] = true;
  }
  axes_ = new_axes;
  return static_cast<Tensor &>(*this);
}
template <typename T, int Dim>
ITensor<T, Dim>::Tensor &ITensor<T, Dim>::transpose(int axis_a, int axis_b) {
  checkAxisInDim(axis_a);
  checkAxisInDim(axis_b);
  if (axis_a == axis_b)
    throw std::invalid_argument("Duplicate axis index");
  std::swap(axes_[axis_a], axes_[axis_b]);
  return static_cast<Tensor &>(*this);
}
template <typename T, int Dim> ITensor<T, Dim>::Tensor &ITensor<T, Dim>::t() {
  static_assert(Dim >= 2, "Can't change the only axis");
  std::swap(axes_[Dim - 1], axes_[Dim - 2]);
  return static_cast<Tensor &>(*this);
}

// ===== OPERATORS ======
template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator+(const T scalar) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result += scalar;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor &ITensor<T, Dim>::operator-=(const T scalar) {
  *this += -scalar;
  return static_cast<Tensor &>(*this);
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator-(const T scalar) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result -= scalar;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator*(const T scalar) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result *= scalar;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor &ITensor<T, Dim>::operator/=(const T scalar) {
  *this *= T(1) / scalar;
  return static_cast<Tensor &>(*this);
}
template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator/(const T scalar) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result /= scalar;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator+(const Tensor &other) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result += other;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor &ITensor<T, Dim>::operator-=(const Tensor &other) {
  checkItHasSameShape(other);
  *this += -other;
  return static_cast<Tensor &>(*this);
}
template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator-(const Tensor &other) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result -= other;
  return result;
}

template <typename T, int Dim>
ITensor<T, Dim>::Tensor ITensor<T, Dim>::operator*(const Tensor &other) const {
  Tensor result = static_cast<const Tensor &>(*this);
  result *= other;
  return result;
}
