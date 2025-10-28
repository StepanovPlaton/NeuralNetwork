#ifndef MATRIX_H
#define MATRIX_H

#include "./opencl/opencl.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

class IMatrix {
protected:
  int rows;
  int cols;

  void validateDimensions(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
      throw std::invalid_argument("Matrix dimensions must be positive");
    }
  }

  void checkIndices(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      throw std::out_of_range("Matrix indices out of range");
    }
  }

public:
  IMatrix(int rows, int cols) : rows(rows), cols(cols) {}
  virtual ~IMatrix() = default;
  virtual int getRows() const = 0;
  virtual int getCols() const = 0;
  virtual const std::vector<float> toVector() const = 0;
};

namespace Matrices {
class CPU;

class GPU : public IMatrix {
protected:
  cl::Buffer *buffer;
  cl::CommandQueue queue;

public:
  GPU(int rows, int cols, const std::vector<float> &matrix)
      : IMatrix(rows, cols), queue(openCL.getContext(), openCL.getDevice()) {
    validateDimensions(rows, cols);
    if (matrix.size() != static_cast<size_t>(rows * cols)) {
      throw std::invalid_argument("Matrix data size doesn't match dimensions");
    }

    buffer = new cl::Buffer(
        openCL.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        rows * cols * sizeof(float), const_cast<float *>(matrix.data()));
  }
  ~GPU() { delete buffer; }

  GPU(const GPU &) = delete;
  GPU &operator=(const GPU &) = delete;
  GPU(GPU &&other) = default;
  GPU &operator=(GPU &&other) = default;

  int getRows() const override { return rows; }
  int getCols() const override { return cols; }
  size_t getSize() const { return rows * cols; }

  const cl::Buffer *getBuffer() const { return buffer; }

  const std::vector<float> toVector() const {
    std::vector<float> result(rows * cols);
    queue.enqueueReadBuffer(*buffer, CL_TRUE, 0, rows * cols * sizeof(float),
                            result.data());
    queue.finish();
    return result;
  }

  CPU toCPU() const;
};

class CPU : public IMatrix {
protected:
  std::vector<float> data;

public:
  CPU(int rows, int cols, float value = 0.0f)
      : IMatrix(rows, cols), data(rows * cols, value) {
    validateDimensions(rows, cols);
  }

  CPU(int rows, int cols, const std::vector<float> &matrix)
      : IMatrix(rows, cols), data(matrix) {
    validateDimensions(rows, cols);
    if (matrix.size() != static_cast<size_t>(rows * cols)) {
      throw std::invalid_argument("Data size doesn't match matrix dimensions");
    }
  }

  CPU(const CPU &) = default;
  CPU &operator=(const CPU &) = default;
  CPU(CPU &&) = default;
  CPU &operator=(CPU &&) = default;
  ~CPU() override = default;

  float &operator()(int row, int col) {
    checkIndices(row, col);
    return data[row * cols + col];
  }

  const float &operator()(int row, int col) const {
    checkIndices(row, col);
    return data[row * cols + col];
  }

  const std::vector<float> toVector() const { return data; }

  int getRows() const override { return rows; }
  int getCols() const override { return cols; }
  size_t getSize() const { return data.size(); }

  GPU toGPU(OpenCL &openCL) const { return GPU(rows, cols, data); }
};

CPU GPU::toCPU() const { return CPU(rows, cols, toVector()); }

} // namespace Matrices

#endif