#pragma once

#include "../../opencl/opencl.hpp"

#include "tensor.hpp"

#include "../math.hpp"

namespace GPU {
template <ITensorType T> class TensorMath;
class Tensor0Math;
class Tensor1Math;
class Tensor2Math;
class Tensor3Math;

template <ITensorType T> class TensorMath : public ITensorMath<T> {
protected:
  enum class Method {
    MULT,
    MULT_SMALL,
    SCALAR_MULT,
    ADD,
    SCALAR_ADD,
    ACTIVATE
  };
  std::unordered_map<Method, cl::Kernel> kernels;
  std::unordered_map<Method, std::string> kernelsNames = {
      {Method::MULT, "mult"},           {Method::MULT_SMALL, "mult_small"},
      {Method::SCALAR_MULT, "mult_sc"}, {Method::ADD, "add"},
      {Method::SCALAR_ADD, "add_sc"},   {Method::ACTIVATE, "activate"}};

  cl::CommandQueue queue;

public:
  TensorMath() {
    queue = cl::CommandQueue(openCL.getContext(), openCL.getDevice());
    for (const auto &entry : kernelsNames) {
      kernels[entry.first] =
          cl::Kernel(openCL.getProgram(OpenCL::Program::MATRIX), entry.second);
    }
  }

  const cl::CommandQueue &getQueue() const { return queue; }

  void await() const override { queue.finish(); }

  T activate(const T &t, Activation type = Activation::LINEAR,
             float alpha = 0.0f) override {
    T result(t.getShape(), false, &queue);
    kernels[Method::ACTIVATE].setArg(0, *t.getBuffer());
    kernels[Method::ACTIVATE].setArg(1, *result.getBuffer());
    kernels[Method::ACTIVATE].setArg(2, static_cast<int>(type));
    kernels[Method::ACTIVATE].setArg(3, alpha);
    queue.enqueueNDRangeKernel(kernels[Method::ACTIVATE], cl::NullRange,
                               cl::NDRange(t.getSize()));
    return result;
  }

  T mult(const T &t, float x) override {
    T result(t.getShape(), false, &queue);
    kernels[Method::SCALAR_MULT].setArg(0, *t.getBuffer());
    kernels[Method::SCALAR_MULT].setArg(1, *result.getBuffer());
    kernels[Method::SCALAR_MULT].setArg(2, x);
    queue.enqueueNDRangeKernel(kernels[Method::SCALAR_MULT], cl::NullRange,
                               cl::NDRange(t.getSize()));
    return result;
  }

  T add(const T &a, const T &b, float x = 1.0f) override {
    this->validateSameDimensions(a, b);
    T result(a.getShape(), false, &queue);
    kernels[Method::ADD].setArg(0, *a.getBuffer());
    kernels[Method::ADD].setArg(1, *b.getBuffer());
    kernels[Method::ADD].setArg(2, *result.getBuffer());
    kernels[Method::ADD].setArg(3, x);
    queue.enqueueNDRangeKernel(kernels[Method::ADD], cl::NullRange,
                               cl::NDRange(a.getSize()));
    return result;
  }

  T add(const T &t, float x) override {
    T result(t.getShape(), false, &queue);
    kernels[Method::SCALAR_ADD].setArg(0, *t.getBuffer());
    kernels[Method::SCALAR_ADD].setArg(1, *result.getBuffer());
    kernels[Method::SCALAR_ADD].setArg(2, x);
    queue.enqueueNDRangeKernel(kernels[Method::SCALAR_ADD], cl::NullRange,
                               cl::NDRange(t.getSize()));
    return result;
  }
};

class Tensor0Math : public TensorMath<Tensor0>, public ITensor0Math<Tensor0> {};

class Tensor1Math : public TensorMath<Tensor1>, public ITensor1Math<Tensor1> {};

class Tensor2Math : public TensorMath<Tensor2>,
                    public ITensor2Math<Tensor2, Tensor1> {
private:
  Tensor2 mult_tiled(const Tensor2 &a, const Tensor2 &b, bool transpose,
                     const Vector &bias, Activation type, float alpha) {
    Tensor2 result(a.getRows(), transpose ? b.getRows() : b.getCols(), false,
                   &queue);

    const int tile_size = 16;
    cl::NDRange local_size(tile_size, tile_size);
    cl::NDRange global_size(
        ((result.getRows() + tile_size - 1) / tile_size) * tile_size,
        ((result.getCols() + tile_size - 1) / tile_size) * tile_size);

    kernels[Method::MULT].setArg(0, *a.getBuffer());
    kernels[Method::MULT].setArg(1, *b.getBuffer());
    kernels[Method::MULT].setArg(2, *result.getBuffer());
    kernels[Method::MULT].setArg(3, *bias.getBuffer());
    kernels[Method::MULT].setArg(4, static_cast<int>(type));
    kernels[Method::MULT].setArg(5, alpha);
    kernels[Method::MULT].setArg(6, result.getRows());
    kernels[Method::MULT].setArg(7, result.getCols());
    kernels[Method::MULT].setArg(8, a.getCols());
    kernels[Method::MULT].setArg(9, transpose ? 1 : 0);
    queue.enqueueNDRangeKernel(kernels[Method::MULT], cl::NullRange,
                               global_size, local_size);
    return result;
  }
  Tensor2 mult_small(const Tensor2 &a, const Tensor2 &b, bool transpose,
                     const Vector &bias, Activation type, float alpha) {
    Tensor2 result(a.getRows(), transpose ? b.getRows() : b.getCols(), false,
                   &queue);
    kernels[Method::MULT_SMALL].setArg(0, *a.getBuffer());
    kernels[Method::MULT_SMALL].setArg(1, *b.getBuffer());
    kernels[Method::MULT_SMALL].setArg(2, *result.getBuffer());
    kernels[Method::MULT_SMALL].setArg(3, *bias.getBuffer());
    kernels[Method::MULT_SMALL].setArg(4, static_cast<int>(type));
    kernels[Method::MULT_SMALL].setArg(5, alpha);
    kernels[Method::MULT_SMALL].setArg(6, result.getRows());
    kernels[Method::MULT_SMALL].setArg(7, result.getCols());
    kernels[Method::MULT_SMALL].setArg(8, a.getCols());
    kernels[Method::MULT_SMALL].setArg(9, transpose ? 1 : 0);
    queue.enqueueNDRangeKernel(kernels[Method::MULT_SMALL], cl::NullRange,
                               cl::NDRange(result.getRows(), result.getCols()));
    return result;
  }

public:
  Tensor2 mult(const Tensor2 &a, const Tensor2 &b, bool transpose = false,
               const Vector *bias = nullptr,
               Activation type = Activation::LINEAR,
               float alpha = 0.01f) override {
    validateMultDimensions(a, b, transpose);
    const Vector defaultBias(a.getRows(), 0.0f, &queue);
    if (bias != nullptr)
      validateBiasDimensions(b, *bias, transpose);
    if (a.getRows() > 64 || a.getCols() > 64 || b.getRows() > 64 ||
        b.getCols() > 64)
      return mult_tiled(a, b, transpose, bias == nullptr ? defaultBias : *bias,
                        type, alpha);
    else
      return mult_small(a, b, transpose, bias == nullptr ? defaultBias : *bias,
                        type, alpha);
  }
};

class Tensor3Math : public TensorMath<Tensor3>, public ITensor3Math<Tensor3> {};

typedef Tensor0Math ScalarMath;
typedef Tensor1Math VectorMath;
typedef Tensor2Math MatrixMath;

} // namespace GPU
