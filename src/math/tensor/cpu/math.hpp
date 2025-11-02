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
  float activateX(float x, Activation type, float alpha = 0.01f) {
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
    default:
      throw std::invalid_argument("Unknown activation type");
    }
  }
  float d_activateX(float x, Activation type, float alpha = 0.01f) {
    switch (type) {
    case Activation::LINEAR:
      return 1.0f;
    case Activation::SIGMOID: {
      float sigmoid = 1.0f / (1.0f + std::exp(-x));
      return sigmoid * (1.0f - sigmoid);
    }
    case Activation::TANH: {
      float tanh_x = std::tanh(x);
      return 1.0f - tanh_x * tanh_x;
    }
    case Activation::RELU:
      return (x > 0.0f) ? 1.0f : 0.0f;
    case Activation::LEAKY_RELU:
      return (x > 0.0f) ? 1.0f : alpha;
    case Activation::ELU:
      return (x > 0.0f) ? 1.0f : alpha * std::exp(x);
    default:
      throw std::invalid_argument("Unknown activation type");
    }
  }

public:
  T activate(const T &t, Activation type = Activation::LINEAR,
             float alpha = 0.0f) override {
    T result(t.getShape(), false);
    for (size_t i = 0; i < t.getSize(); ++i) {
      result[i] = activateX(t[i], type, alpha);
    }
    return result;
  }
  T d_activate(const T &t, Activation type = Activation::LINEAR,
               float alpha = 0.0f) override {
    T result(t.getShape(), false);
    for (size_t i = 0; i < t.getSize(); ++i) {
      result[i] = d_activateX(t[i], type, alpha);
    }
    return result;
  }

  T mult(const T &a, const T &b) override {
    this->validateSameDimensions(a, b);
    T result(a.getShape(), false);
    for (size_t i = 0; i < a.getSize(); ++i)
      result[i] = a[i] * b[i];
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

  void await() const override {}
};

class Tensor0Math : public TensorMath<Tensor0>, public ITensor0Math<Tensor0> {};

class Tensor1Math : public TensorMath<Tensor1>, public ITensor1Math<Tensor1> {};

class Tensor2Math : public TensorMath<Tensor2>,
                    public ITensor2Math<Tensor2, Tensor1> {
private:
  Tensor2 mse(const Tensor2 &a, const Tensor2 &b) {
    Tensor2 result(a.getShape(), false);
    for (size_t i = 0; i < result.getSize(); ++i)
      result[i] += (a[i] - b[i]) * (a[i] - b[i]) / (float)a.getCols();
    return result;
  }
  Tensor2 dmse(const Tensor2 &a, const Tensor2 &b) {
    Tensor2 result(a.getShape(), false);
    for (size_t i = 0; i < result.getSize(); ++i)
      result[i] += 2 * (a[i] - b[i]) / (float)a.getCols();
    return result;
  }

public:
  Tensor2 dot(const Tensor2 &a, const Tensor2 &b, bool transpose_a = false,
              bool transpose_b = false, const Vector *bias = nullptr,
              Activation type = Activation::LINEAR,
              float alpha = 0.01f) override {
    validateMultDimensions(a, b, transpose_a, transpose_b);
    if (bias != nullptr)
      validateBiasDimensions(b, *bias, transpose_b);
    Tensor2 result(transpose_a ? a.getCols() : a.getRows(),
                   transpose_b ? b.getRows() : b.getCols(), 0.0f);
    for (int i = 0; i < result.getRows(); ++i) {
      for (int j = 0; j < result.getCols(); ++j) {
        float sum = 0.0f;
        for (int k = 0; k < a.getCols(); ++k)
          sum += (transpose_a ? a(k, i) : a(i, k)) *
                 (transpose_b ? b(j, k) : b(k, j));
        result(i, j) =
            activateX(sum + (bias == nullptr ? 0.0f : (*bias)(j)), type, alpha);
      }
    }
    return result;
  }

  Tensor2 loss(const Tensor2 &a, const Tensor2 &b, Loss type) override {
    this->validateSameDimensions(a, b);
    switch (type) {
    case Loss::MSE:
      return mse(a, b);
    default:
      throw std::invalid_argument("Unknown loss type");
    }
  }
  Tensor2 d_loss(const Tensor2 &a, const Tensor2 &b, Loss type) override {
    this->validateSameDimensions(a, b);
    switch (type) {
    case Loss::MSE:
      return dmse(a, b);
    default:
      throw std::invalid_argument("Unknown loss type");
    }
  }

  Tensor1 axis_sum(const Tensor2 &m) override {
    Tensor1 result(m.getCols(), 0.0f);
    for (int i = 0; i < m.getCols(); ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m.getRows(); ++j)
        sum += m(j, i);
      result(i) = sum;
    }
    return result;
  }
};

class Tensor3Math : public TensorMath<Tensor3>, public ITensor3Math<Tensor3> {};

typedef Tensor0Math ScalarMath;
typedef Tensor1Math VectorMath;
typedef Tensor2Math MatrixMath;

} // namespace CPU
