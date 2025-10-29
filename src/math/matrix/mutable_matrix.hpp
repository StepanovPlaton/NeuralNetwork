#pragma once

#include "matrix.hpp"

template <typename T> class IMutableMatrix {
  static_assert(std::is_base_of<IMatrix, T>::value,
                "T must be derived from IMatrix");

public:
  enum class Activate { LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, ELU, GELU };

  virtual void mult(T &m, float bias, Activate type, float alpha) = 0;
  virtual void mult(float s) = 0;
  virtual void add(T &m, float a, float b) = 0;
  virtual void add(float a) = 0;
  virtual void activate(Activate type, float alpha = 0.01f) = 0;

  void validateMultDimensions(T &a, T &b) const {
    if (a.getRows() != b.getCols()) {
      throw std::invalid_argument(
          "Invalid matrix dimensions for multiplication");
    }
  };
  void validateSameDimensions(T &a, T &b) const {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
      throw std::invalid_argument("Invalid matrix dimensions for addition");
    }
  };
};
