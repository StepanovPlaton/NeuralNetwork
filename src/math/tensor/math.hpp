#pragma once

#include "tensor.hpp"

enum class Activation { LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, ELU, GELU };

template <typename T>
concept ITensorType = std::is_base_of_v<ITensor, T>;

template <typename T>
concept ITensor0Type = std::is_base_of_v<ITensor0, T>;
template <typename T>
concept ITensor1Type = std::is_base_of_v<ITensor1, T>;
template <typename T>
concept ITensor2Type = std::is_base_of_v<ITensor2, T>;
template <typename T>
concept ITensor3Type = std::is_base_of_v<ITensor3, T>;

template <ITensorType T> class ITensorMath {
protected:
  void validateSameDimensions(const T &a, const T &b) const {
    if (a.getDim() != b.getDim())
      throw std::invalid_argument("Tensors must have the same dimension");
    if (a.getSize() != b.getSize())
      throw std::invalid_argument("Tensors must have the same size");
    for (int i = 0; i < a.getDim(); ++i) {
      if (a.getShape()[i] != b.getShape()[i])
        throw std::invalid_argument("Tensors must have the same shape");
    }
  };

public:
  virtual T activate(const T &m, Activation type, float alpha) = 0;

  virtual T mult(const T &m, float x) = 0;
  virtual T add(const T &a, const T &b, float x) = 0;
  virtual T add(const T &m, float x) = 0;
};

template <ITensor0Type T> class ITensor0Math {};

template <ITensor1Type T> class ITensor1Math {};

template <ITensor2Type M, ITensor1Type V> class ITensor2Math {
public:
  virtual M mult(const M &a, const M &b, bool transpose, const V *bias,
                 Activation type, float alpha) = 0;

  void validateMultDimensions(const M &a, const M &b, bool transpose) const {
    if ((!transpose && a.getCols() != b.getRows()) ||
        (transpose && a.getCols() != b.getCols())) {
      throw std::invalid_argument(
          "Invalid matrix dimensions for multiplication");
    }
  };
  void validateBiasDimensions(const M &a, const V &b, bool transpose) const {
    if ((!transpose && a.getCols() != b.getSize()) ||
        (transpose && a.getRows() != b.getSize())) {
      throw std::invalid_argument("Invalid matrix bias");
    }
  };
};

template <ITensor3Type T> class ITensor3Math {};