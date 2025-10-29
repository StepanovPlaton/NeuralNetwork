#include "matrix.hpp"

Matrices::CPU::CPU(int rows, int cols, float value)
    : IMatrix(rows, cols), data(rows * cols, value) {
  validateDimensions(rows, cols);
}

Matrices::CPU::CPU(int rows, int cols, const std::vector<float> &matrix)
    : IMatrix(rows, cols), data(matrix) {
  validateDimensions(rows, cols);
  if (matrix.size() != static_cast<size_t>(rows * cols)) {
    throw std::invalid_argument("Data size doesn't match matrix dimensions");
  }
}

float &Matrices::CPU::operator()(int row, int col) {
  checkIndices(row, col);
  return data[row * cols + col];
}

const float &Matrices::CPU::operator()(int row, int col) const {
  checkIndices(row, col);
  return data[row * cols + col];
}