#include "mutable_matrix.hpp"

float MutableMatrices::CPU::activate_x(float x, Activate type, float alpha) {
  switch (type) {
  case Activate::LINEAR:
    return x;
  case Activate::SIGMOID:
    return 1.0f / (1.0f + std::exp(-x));
  case Activate::TANH:
    return std::tanh(x);
  case Activate::RELU:
    return std::max(0.0f, x);
  case Activate::LEAKY_RELU:
    return (x > 0.0f) ? x : alpha * x;
  case Activate::ELU:
    return (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
  case Activate::GELU:
    return 0.5f * x *
           (1.0f +
            std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
  default:
    throw std::invalid_argument("Unknown activation type");
  }
}
void MutableMatrices::CPU::mult(Matrices::CPU &m, float bias, Activate type,
                                float alpha) {
  validateMultDimensions(*this, m);

  std::vector<float> result(rows * m.getCols(), 0.0f);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < m.getCols(); j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols; k++) {
        sum += (*this)(i, k) * m(k, j);
      }
      result[i * m.getCols() + j] = activate_x(sum + bias, type, alpha);
    }
  }
  data = std::move(result);
  cols = m.getCols();
}

void MutableMatrices::CPU::mult(float scalar) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data[i * cols + j] *= scalar;
    }
  }
}

void MutableMatrices::CPU::add(Matrices::CPU &m, float a, float b) {
  validateSameDimensions(*this, m);

  std::vector<float> result(rows * cols, 0.0f);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i * cols + j] = ((*this)(i, j) * a) + (m(i, j) * b);
    }
  }
  data = std::move(result);
}

void MutableMatrices::CPU::add(float scalar) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data[i * cols + j] += scalar;
    }
  }
}
void MutableMatrices::CPU::activate(Activate type, float alpha) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data[i * cols + j] = activate_x(data[i * cols + j], type, alpha);
    }
  }
}