#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>

#include "device.hpp"

class Matrix {
protected:
  cl_mem buf;
  size_t rows;
  size_t cols;

public:
  Matrix(CalcEngine &calcEngine, cl_mem_flags flags, size_t rows, size_t cols,
         float *matrix)
      : rows(rows), cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Размеры матрицы должны быть больше 0");
    }
    buf = calcEngine.createBuffer(flags, rows * cols * sizeof(float), matrix);
  }

  ~Matrix() { clReleaseMemObject(buf); }

  size_t getRows() const { return rows; }
  size_t getCols() const { return cols; }

  const cl_mem getBuf() const { return buf; }
};

#endif
