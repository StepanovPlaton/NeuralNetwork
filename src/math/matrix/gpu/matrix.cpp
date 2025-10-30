#include <random>

#include "matrix.hpp"

std::random_device rd;
std::mt19937 gen(rd());

Matrices::GPU::GPU(int rows, int cols)
    : IMatrix(rows, cols), queue(openCL.getContext(), openCL.getDevice()) {
  validateDimensions(rows, cols);
  std::vector<float> matrix;
  matrix.reserve(rows * cols);
  for (size_t i = 0; i < (size_t)rows * (size_t)cols; ++i)
    matrix.push_back(std::generate_canonical<float, 32>(gen));
  buffer = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                          rows * cols * sizeof(float));
  queue.enqueueWriteBuffer(*buffer, CL_TRUE, 0, rows * cols * sizeof(float),
                           matrix.data());
  queue.finish();
}

Matrices::GPU::GPU(int rows, int cols, const std::vector<float> &matrix)
    : IMatrix(rows, cols), queue(openCL.getContext(), openCL.getDevice()) {
  validateDimensions(rows, cols);
  if (matrix.size() != static_cast<size_t>(rows * cols)) {
    throw std::invalid_argument("Matrix data size doesn't match dimensions");
  }
  buffer = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                          rows * cols * sizeof(float));
  queue.enqueueWriteBuffer(*buffer, CL_TRUE, 0, rows * cols * sizeof(float),
                           matrix.data());
  queue.finish();
}

const std::vector<float> Matrices::GPU::toVector() const {
  std::vector<float> result(rows * cols);
  queue.enqueueReadBuffer(*buffer, CL_TRUE, 0, rows * cols * sizeof(float),
                          result.data());
  queue.finish();
  return result;
}