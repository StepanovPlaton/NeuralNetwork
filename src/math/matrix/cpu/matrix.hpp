#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../matrix.hpp"

namespace Matrices {

class CPU : public IMatrix {
protected:
  std::vector<float> data;

public:
  CPU(int rows, int cols, float value = 0.0f);
  CPU(int rows, int cols, const std::vector<float> &matrix);

  CPU(const CPU &) = default;
  CPU &operator=(const CPU &) = default;
  CPU(CPU &&) = default;
  CPU &operator=(CPU &&) = default;
  ~CPU() override = default;

  float &operator()(int row, int col);
  const float &operator()(int row, int col) const;

  const std::vector<float> toVector() const { return data; }

  int getRows() const override { return rows; }
  int getCols() const override { return cols; }
  size_t getSize() const { return data.size(); }

  //   GPU toGPU(OpenCL &openCL) const { return GPU(rows, cols, data); }
};

} // namespace Matrices
