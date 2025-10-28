#ifndef MUTABLE_MATRIX_H
#define MUTABLE_MATRIX_H

#include "./opencl/opencl.hpp"

#include "matrix.hpp"

template <typename T> class IMutableMatrix {
  static_assert(std::is_base_of<IMatrix, T>::value,
                "T must be derived from IMatrix");

public:
  virtual void mult(T &m) = 0;
  virtual void mult(float s) = 0;
  virtual void add(T &m, float a, float b) = 0;
  virtual void add(float a) = 0;

  void validateMultDimensions(T &a, T &b) {
    if (a.getRows() != b.getCols()) {
      throw std::invalid_argument(
          "Invalid matrix dimensions for multiplication");
    }
  }
  void validateSameDimensions(T &a, T &b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
      throw std::invalid_argument("Invalid matrix dimensions for addition");
    }
  }
};

namespace MutableMatrices {
class GPU : public Matrices::GPU, public IMutableMatrix<Matrices::GPU> {
private:
  enum class Method { MULT, SCALAR_MULT, ADD, SCALAR_ADD };
  std::unordered_map<Method, cl::Kernel> kernels;
  std::unordered_map<Method, std::string> kernelsNames = {
      {Method::MULT, "mult"},
      {Method::SCALAR_MULT, "mult_sc"},
      {Method::ADD, "add"},
      {Method::SCALAR_ADD, "add_sc"}};

  static void CL_CALLBACK releaseBuffer(cl_event event, cl_int status,
                                        void *buf) {
    if (status == CL_COMPLETE) {
      //   std::cout << "Kernel complete!" << std::endl;
      delete buf;
    }
  }

public:
  GPU(int rows, int cols, const std::vector<float> &matrix)
      : Matrices::GPU(rows, cols, matrix) {
    for (const auto &[method, kernelName] : kernelsNames) {
      kernels[method] =
          cl::Kernel(openCL.getProgram(OpenCL::Program::MATRIX), kernelName);
    }
  }

  void mult(Matrices::GPU &m) {
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
    kernels[Method::MULT].setArg(3, rows);
    kernels[Method::MULT].setArg(4, m.getCols());
    kernels[Method::MULT].setArg(5, cols);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernels[Method::MULT], cl::NullRange,
                               global_size, local_size, nullptr, &event);

    event.setCallback(CL_COMPLETE, releaseBuffer, buffer);
    buffer = b;
    cols = m.getCols();
  }

  void mult(float scalar) {
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

  void add(Matrices::GPU &m, float a = 1.0f, float b = 1.0f) {
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

  void add(float scalar) {
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
};
class CPU : public Matrices::CPU, public IMutableMatrix<Matrices::CPU> {

public:
  CPU(int rows, int cols, const std::vector<float> &matrix)
      : Matrices::CPU(rows, cols, matrix) {}

  void mult(Matrices::CPU &m) {
    validateMultDimensions(*this, m);

    std::vector<float> result(rows * m.getCols(), 0.0f);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < m.getCols(); j++) {
        float sum = 0.0f;
        for (int k = 0; k < cols; k++) {
          sum += (*this)(i, k) * m(k, j);
        }
        result[i * m.getCols() + j] = sum;
      }
    }
    data = std::move(result);
    cols = m.getCols();
  }

  void mult(float scalar) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i * cols + j] *= scalar;
      }
    }
  }

  void add(Matrices::CPU &m, float a = 1.0f, float b = 1.0f) {
    validateSameDimensions(*this, m);

    std::vector<float> result(rows * cols, 0.0f);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i * cols + j] = ((*this)(i, j) * a) + (m(i, j) * b);
      }
    }
    data = std::move(result);
  }

  void add(float scalar) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i * cols + j] += scalar;
      }
    }
  }
};
}; // namespace MutableMatrices

#endif