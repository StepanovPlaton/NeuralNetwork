#include "matrix.hpp"

Matrices::GPU::GPU(int rows, int cols, const std::vector<float> &matrix)
    : IMatrix(rows, cols), queue(openCL.getContext(), openCL.getDevice()) {
  validateDimensions(rows, cols);
  if (matrix.size() != static_cast<size_t>(rows * cols)) {
    throw std::invalid_argument("Matrix data size doesn't match dimensions");
  }

  buffer = new cl::Buffer(
      openCL.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      rows * cols * sizeof(float), const_cast<float *>(matrix.data()));
}

const std::vector<float> Matrices::GPU::toVector() const {
  std::vector<float> result(rows * cols);
  queue.enqueueReadBuffer(*buffer, CL_TRUE, 0, rows * cols * sizeof(float),
                          result.data());
  queue.finish();
  return result;
}