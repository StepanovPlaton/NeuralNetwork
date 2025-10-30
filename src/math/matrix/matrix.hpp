#pragma once

#include <random>
#include <stdexcept>
#include <vector>

class IMatrix {
protected:
  int rows;
  int cols;

  void validateDimensions(int rows, int cols) const {
    if (rows <= 0 || cols <= 0) {
      throw std::invalid_argument("Matrix dimensions must be positive");
    }
  };
  void checkIndices(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      throw std::out_of_range("Matrix indices out of range");
    }
  };

public:
  IMatrix(int rows, int cols) : rows(rows), cols(cols) {}
  virtual ~IMatrix() = default;
  virtual int getRows() const = 0;
  virtual int getCols() const = 0;
  virtual const std::vector<float> toVector() const = 0;
};
