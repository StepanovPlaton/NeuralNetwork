#pragma once

#include "../../opencl/opencl.hpp"

#include "../matrix.hpp"

namespace Matrices {
class GPU : public IMatrix {
protected:
  cl::Buffer *buffer;
  cl::CommandQueue queue;

public:
  GPU(int rows, int cols);
  GPU(int rows, int cols, const std::vector<float> &matrix);
  ~GPU() { delete buffer; }

  GPU(const GPU &) = delete;
  GPU &operator=(const GPU &) = delete;
  GPU(GPU &&other)
      : IMatrix(other.rows, other.cols), buffer(other.buffer),
        queue(std::move(other.queue)) {
    other.buffer = nullptr;
    other.rows = 0;
    other.cols = 0;
  }
  GPU &operator=(GPU &&other) = default;

  int getRows() const override { return rows; }
  int getCols() const override { return cols; }
  size_t getSize() const { return rows * cols; }

  const cl::Buffer *getBuffer() const { return buffer; }

  const std::vector<float> toVector() const;

  // CPU toCPU() const { return CPU(rows, cols, toVector()); };
};

} // namespace Matrices
