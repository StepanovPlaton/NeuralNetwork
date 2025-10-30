#include "mutable_matrix.hpp"

MutableMatrices::GPU::GPU(int rows, int cols) : Matrices::GPU(rows, cols) {
  for (const auto &entry : kernelsNames) {
    kernels[entry.first] =
        cl::Kernel(openCL.getProgram(OpenCL::Program::MATRIX), entry.second);
  }
}

MutableMatrices::GPU::GPU(int rows, int cols, const std::vector<float> &matrix)
    : Matrices::GPU(rows, cols, matrix) {
  for (const auto &entry : kernelsNames) {
    kernels[entry.first] =
        cl::Kernel(openCL.getProgram(OpenCL::Program::MATRIX), entry.second);
  }
}

void MutableMatrices::GPU::mult(Matrices::GPU &m, float bias, Activate type,
                                float alpha) {
  validateMultDimensions(*this, m);

  cl::Buffer *b = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                                 rows * m.getCols() * sizeof(float));

  const int tile_size = 16;
  cl::NDRange local_size(tile_size, tile_size);
  cl::NDRange global_size(((rows + tile_size - 1) / tile_size) * tile_size,
                          ((m.getCols() + tile_size - 1) / tile_size) *
                              tile_size);

  kernels[Method::MULT].setArg(0, *buffer);
  kernels[Method::MULT].setArg(1, *m.getBuffer());
  kernels[Method::MULT].setArg(2, *b);
  kernels[Method::MULT].setArg(3, bias);
  kernels[Method::MULT].setArg(4, static_cast<int>(type));
  kernels[Method::MULT].setArg(5, alpha);
  kernels[Method::MULT].setArg(6, rows);
  kernels[Method::MULT].setArg(7, m.getCols());
  kernels[Method::MULT].setArg(8, cols);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernels[Method::MULT], cl::NullRange, global_size,
                             local_size, nullptr, &event);

  event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
  buffer = b;
  cols = m.getCols();
}

void MutableMatrices::GPU::mult(float scalar) {
  cl::Buffer *b = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                                 rows * cols * sizeof(float));
  kernels[Method::SCALAR_MULT].setArg(0, *buffer);
  kernels[Method::SCALAR_MULT].setArg(1, *b);
  kernels[Method::SCALAR_MULT].setArg(2, scalar);
  kernels[Method::SCALAR_MULT].setArg(3, rows);
  kernels[Method::SCALAR_MULT].setArg(4, cols);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernels[Method::SCALAR_MULT], cl::NullRange,
                             cl::NDRange(rows, cols), cl::NullRange, nullptr,
                             &event);

  event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
  buffer = b;
}

void MutableMatrices::GPU::add(Matrices::GPU &m, float a, float b) {
  validateSameDimensions(*this, m);

  cl::Buffer *buf = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                                   rows * cols * sizeof(float));
  kernels[Method::ADD].setArg(0, *buffer);
  kernels[Method::ADD].setArg(1, *m.getBuffer());
  kernels[Method::ADD].setArg(2, *buf);
  kernels[Method::ADD].setArg(3, a);
  kernels[Method::ADD].setArg(4, b);
  kernels[Method::ADD].setArg(5, rows);
  kernels[Method::ADD].setArg(6, cols);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernels[Method::ADD], cl::NullRange,
                             cl::NDRange(rows, cols), cl::NullRange, nullptr,
                             &event);

  event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
  buffer = buf;
}

void MutableMatrices::GPU::add(float scalar) {
  cl::Buffer *b = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                                 rows * cols * sizeof(float));
  kernels[Method::SCALAR_ADD].setArg(0, *buffer);
  kernels[Method::SCALAR_ADD].setArg(1, *b);
  kernels[Method::SCALAR_ADD].setArg(2, scalar);
  kernels[Method::SCALAR_ADD].setArg(3, rows);
  kernels[Method::SCALAR_ADD].setArg(4, cols);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernels[Method::SCALAR_ADD], cl::NullRange,
                             cl::NDRange(rows, cols), cl::NullRange, nullptr,
                             &event);

  event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
  buffer = b;
}

void MutableMatrices::GPU::activate(Activate type, float alpha) {
  cl::Buffer *b = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                                 rows * cols * sizeof(float));
  kernels[Method::ACTIVATE].setArg(0, *buffer);
  kernels[Method::ACTIVATE].setArg(1, *b);
  kernels[Method::ACTIVATE].setArg(2, static_cast<int>(type));
  kernels[Method::ACTIVATE].setArg(3, alpha);
  kernels[Method::ACTIVATE].setArg(4, rows);
  kernels[Method::ACTIVATE].setArg(5, cols);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernels[Method::ACTIVATE], cl::NullRange,
                             cl::NDRange(rows, cols), cl::NullRange, nullptr,
                             &event);

  event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
  buffer = b;
}