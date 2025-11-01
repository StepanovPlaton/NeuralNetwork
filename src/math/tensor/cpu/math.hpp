#pragma once

#include "tensor.hpp"

#include "../math.hpp"

#include <cmath>

#define M_PI 3.14159265358979323846

namespace CPU {
template <ITensorType T> class TensorMath;
class Tensor0Math;
class Tensor1Math;
class Tensor2Math;
class Tensor3Math;

template <ITensorType T> class TensorMath : public ITensorMath<T> {
protected:
  float activate_x(float x, Activation type, float alpha = 0.01f) {
    switch (type) {
    case Activation::LINEAR:
      return x;
    case Activation::SIGMOID:
      return 1.0f / (1.0f + std::exp(-x));
    case Activation::TANH:
      return std::tanh(x);
    case Activation::RELU:
      return std::max(0.0f, x);
    case Activation::LEAKY_RELU:
      return (x > 0.0f) ? x : alpha * x;
    case Activation::ELU:
      return (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
    case Activation::GELU:
      return 0.5f * x *
             (1.0f +
              std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    default:
      throw std::invalid_argument("Unknown activation type");
    }
  }

public:
  T activate(const T &t, Activation type = Activation::LINEAR,
             float alpha = 0.0f) override {
    T result(t.getShape(), false);
    for (size_t i = 0; i < t.getSize(); ++i) {
      result[i] = activate_x(t[i], type, alpha);
    }
    return result;
  }

  T mult(const T &t, float x) override {
    T result(t.getShape(), false);
    for (size_t i = 0; i < t.getSize(); ++i)
      result[i] = t[i] * x;
    return result;
  }
  T add(const T &a, const T &b, float x = 1.0f) override {
    this->validateSameDimensions(a, b);
    T result(a.getShape(), false);
    for (size_t i = 0; i < a.getSize(); ++i)
      result[i] = a[i] + (b[i] * x);
    return result;
  }
  T add(const T &t, float x) override {
    T result(t.getShape(), false);
    for (size_t i = 0; i < t.getSize(); ++i)
      result[i] = t[i] + x;
    return result;
  }
};

class Tensor0Math : public TensorMath<Tensor0>, public ITensor0Math<Tensor0> {};

class Tensor1Math : public TensorMath<Tensor1>, public ITensor1Math<Tensor1> {};

class Tensor2Math : public TensorMath<Tensor2>, public ITensor2Math<Tensor2> {
public:
  Tensor2 mult(const Tensor2 &a, const Tensor2 &b, bool transpose = false,
               float bias = 0.0f, Activation type = Activation::LINEAR,
               float alpha = 0.01f) override {
    validateMultDimensions(a, b, transpose);
    Tensor2 result(a.getRows(), b.getCols(), 0.0f);
    for (int i = 0; i < result.getRows(); ++i) {
      for (int j = 0; j < result.getCols(); ++j) {
        float sum = 0.0f;
        for (int k = 0; k < a.getCols(); ++k)
          sum += a(i, k) * (transpose ? b(j, k) : b(k, j));
        result(i, j) = activate_x(sum + bias, type, alpha);
      }
    }
    return result;
  }
};

class Tensor3Math : public TensorMath<Tensor3>, public ITensor3Math<Tensor3> {};

typedef Tensor0Math ScalarMath;
typedef Tensor1Math VectorMath;
typedef Tensor2Math MatrixMath;

} // namespace CPU
