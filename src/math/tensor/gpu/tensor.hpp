#pragma once

#include "../../opencl/opencl.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "../tensor.hpp"
#include "math.hpp"

extern std::mt19937 gen;

namespace GPU {
class Tensor;
class Tensor0;
class Tensor1;
class Tensor2;
class Tensor3;

class Tensor : public ITensor {
protected:
  cl::Buffer *buffer = nullptr;

  size_t getShapeSize(const std::vector<int> &shape) {
    size_t size = 1;
    for (int dim : shape)
      size *= dim;
    return size;
  }
  void fillBuf(const std::vector<float> &v,
               const cl::CommandQueue *queue = nullptr) {
    if (buffer != nullptr)
      throw std::runtime_error("Tensor buffer already exists");
    buffer = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                            v.size() * sizeof(float));
    cl::CommandQueue q = queue == nullptr ? openCL.getDefaultQueue() : *queue;
    q.enqueueWriteBuffer(*buffer, CL_TRUE, 0, v.size() * sizeof(float),
                         v.data());
    q.finish();
  }
  void createBuf(size_t size, const cl::CommandQueue *queue = nullptr) {
    std::vector<float> v(size);
    std::generate(v.begin(), v.end(),
                  []() { return std::generate_canonical<float, 10>(gen); });
    fillBuf(v, queue);
  }
  void createBuf(size_t size, float value,
                 const cl::CommandQueue *queue = nullptr) {
    std::vector<float> v(size);
    std::fill(v.begin(), v.end(), value);
    fillBuf(v, queue);
  }

public:
  Tensor(const std::vector<int> &shape, const cl::CommandQueue *queue = nullptr)
      : ITensor(shape) {
    createBuf(getShapeSize(shape), queue);
  }
  Tensor(const std::vector<int> &shape, float value,
         const cl::CommandQueue *queue = nullptr)
      : ITensor(shape) {
    createBuf(getShapeSize(shape), value, queue);
  }
  Tensor(const std::vector<int> &shape, bool fill,
         const cl::CommandQueue *queue = nullptr)
      : ITensor(shape) {
    if (fill)
      createBuf(getShapeSize(shape), 0.0f, queue);
  }

  Tensor(const Tensor &other, const cl::CommandQueue *queue = nullptr)
      : ITensor(other) {
    cl::CommandQueue q = queue == nullptr ? openCL.getDefaultQueue() : *queue;
    createBuf(other.getSize(), &q);
    q.enqueueCopyBuffer(*other.buffer, *buffer, 0, 0,
                        other.getSize() * sizeof(float));
  };
  Tensor &operator=(const Tensor &other) {
    if (buffer != nullptr)
      delete buffer;
    ITensor::operator=(other);
    createBuf(other.getSize(), &openCL.getDefaultQueue());
    openCL.getDefaultQueue().enqueueCopyBuffer(*other.buffer, *buffer, 0, 0,
                                               other.getSize() * sizeof(float));
    return *this;
  };
  Tensor(Tensor &&other) : ITensor(other), buffer(other.buffer) {
    other.buffer = nullptr;
  };
  Tensor &operator=(Tensor &&other) {
    if (this != &other) {
      if (buffer != nullptr)
        delete buffer;
      ITensor::operator=(std::move(other));
      buffer = other.buffer;
      other.buffer = nullptr;
    }
    return *this;
  };

  ~Tensor() {
    if (buffer != nullptr)
      delete buffer;
  }

  std::vector<float> toVector(const cl::CommandQueue *queue = nullptr) {
    size_t size = getShapeSize(shape);
    std::vector<float> result(size);
    cl::CommandQueue q = queue == nullptr ? openCL.getDefaultQueue() : *queue;
    q.enqueueReadBuffer(*buffer, CL_TRUE, 0, size * sizeof(float),
                        result.data());
    q.finish();
    return result;
  }

  const cl::Buffer *getBuffer() const { return buffer; }

  static Tensor0 *asScalar(Tensor *tensor) {
    return tensor->getType() == Type::SCALAR
               ? reinterpret_cast<Tensor0 *>(tensor)
               : nullptr;
  }
  static const Tensor0 *asScalar(const Tensor *tensor) {
    return tensor->getType() == Type::SCALAR
               ? reinterpret_cast<const Tensor0 *>(tensor)
               : nullptr;
  }
  static Tensor1 *asVector(Tensor *tensor) {
    return tensor->getType() == Type::VECTOR
               ? reinterpret_cast<Tensor1 *>(tensor)
               : nullptr;
  }
  static const Tensor1 *asVector(const Tensor *tensor) {
    return tensor->getType() == Type::VECTOR
               ? reinterpret_cast<const Tensor1 *>(tensor)
               : nullptr;
  }
  static Tensor2 *asMatrix(Tensor *tensor) {
    return tensor->getType() == Type::MATRIX
               ? reinterpret_cast<Tensor2 *>(tensor)
               : nullptr;
  }
  static const Tensor2 *asMatrix(const Tensor *tensor) {
    return tensor->getType() == Type::MATRIX
               ? reinterpret_cast<const Tensor2 *>(tensor)
               : nullptr;
  }
  static Tensor3 *asTensor3(Tensor *tensor) {
    return tensor->getType() == Type::TENSOR3
               ? reinterpret_cast<Tensor3 *>(tensor)
               : nullptr;
  }
  static const Tensor3 *asTensor3(const Tensor *tensor) {
    return tensor->getType() == Type::TENSOR3
               ? reinterpret_cast<const Tensor3 *>(tensor)
               : nullptr;
  }
};

class Tensor0 : public Tensor, public ITensor0 {
public:
  Tensor0(const std::vector<int> &shape,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, queue) {
    if (shape.size() != 0)
      throw std::invalid_argument("Tensor0 dimension must be 0");
  }
  Tensor0(const std::vector<int> &shape, float value,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, value, queue) {
    if (shape.size() != 0)
      throw std::invalid_argument("Tensor0 dimension must be 0");
  }
  Tensor0(const cl::CommandQueue *queue = nullptr)
      : Tensor(std::vector<int>{}, queue) {
    createBuf(1, queue);
  }
  Tensor0(float value, const cl::CommandQueue *queue = nullptr)
      : Tensor(std::vector<int>{}, queue) {
    createBuf(1, value, queue);
  }
  Tensor0(const Tensor0 &other, const cl::CommandQueue *queue = nullptr)
      : Tensor(other, queue) {};
  Tensor0 &operator=(const Tensor0 &other) {
    Tensor::operator=(other);
    return *this;
  };
  Tensor0(Tensor0 &&other) : Tensor(std::move(other)) {};
  Tensor0 &operator=(Tensor0 &&other) {
    Tensor::operator=(std::move(other));
    return *this;
  };
};

class Tensor1 : public Tensor, public ITensor1 {
public:
  Tensor1(const std::vector<int> &shape,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, queue) {
    if (shape.size() != 1)
      throw std::invalid_argument("Tensor1 dimension must be 1");
  }
  Tensor1(const std::vector<int> &shape, float value,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, value, queue) {
    if (shape.size() != 1)
      throw std::invalid_argument("Tensor1 dimension must be 1");
  }
  Tensor1(int size, const cl::CommandQueue *queue = nullptr)
      : Tensor({size}, queue) {}
  Tensor1(int size, float value, const cl::CommandQueue *queue = nullptr)
      : Tensor({size}, value, queue) {}
  Tensor1(const std::vector<float> &values,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({(int)values.size()}, false, queue) {
    fillBuf(values, queue);
  }
  Tensor1(const Tensor1 &other, const cl::CommandQueue *queue = nullptr)
      : Tensor(other, queue) {};
  Tensor1 &operator=(const Tensor1 &other) {
    Tensor::operator=(other);
    return *this;
  };
  Tensor1(Tensor1 &&other) : Tensor(std::move(other)) {};
  Tensor1 &operator=(Tensor1 &&other) {
    Tensor::operator=(std::move(other));
    return *this;
  };

  int getSize() const override { return shape[0]; }
};

class Tensor2 : public ITensor2, public Tensor {
public:
  Tensor2(const std::vector<int> &shape,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, queue) {
    if (shape.size() != 2)
      throw std::invalid_argument("Tensor2 dimension must be 2");
  }
  Tensor2(const std::vector<int> &shape, float value,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, value, queue) {
    if (shape.size() != 2)
      throw std::invalid_argument("Tensor2 dimension must be 2");
  }
  Tensor2(int rows, int cols, const cl::CommandQueue *queue = nullptr)
      : ITensor2(), Tensor({rows, cols}, queue) {}
  Tensor2(int rows, int cols, float value,
          const cl::CommandQueue *queue = nullptr)
      : ITensor2(), Tensor({rows, cols}, value, queue) {}
  Tensor2(int rows, int cols, const std::vector<float> &values,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({rows, cols}, false, queue) {
    fillBuf(values, queue);
  }
  Tensor2(const std::vector<std::vector<float>> &values,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({(int)values.size(), (int)values[0].size()}, false) {
    std::vector<float> v(values.size() * values[0].size());
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values[i].size(); ++j)
        v[i * values[0].size() + j] = values[i][j];
    }
    fillBuf(v, queue);
  }

  Tensor2(const Tensor2 &other, const cl::CommandQueue *queue = nullptr)
      : Tensor(other, queue) {};
  Tensor2 &operator=(const Tensor2 &other) {
    Tensor::operator=(other);
    return *this;
  };
  Tensor2(Tensor2 &&other) : Tensor(std::move(other)) {};
  Tensor2 &operator=(Tensor2 &&other) {
    Tensor::operator=(std::move(other));
    return *this;
  };

  int getRows() const override { return shape[0]; }
  int getCols() const override { return shape[1]; }
};

class Tensor3 : public Tensor, public ITensor3 {
public:
  Tensor3(const std::vector<int> &shape,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, queue) {
    if (shape.size() != 3)
      throw std::invalid_argument("Tensor3 dimension must be 3");
  }
  Tensor3(const std::vector<int> &shape, float value,
          const cl::CommandQueue *queue = nullptr)
      : Tensor(shape, value, queue) {
    if (shape.size() != 3)
      throw std::invalid_argument("Tensor3 dimension must be 3");
  }
  Tensor3(int d1, int d2, int d3, const cl::CommandQueue *queue = nullptr)
      : Tensor({d1, d2, d3}, queue) {}
  Tensor3(int d1, int d2, int d3, float value,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({d1, d2, d3}, value, queue) {}
  Tensor3(int d1, int d2, int d3, const std::vector<float> &values,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({d1, d2, d3}, false, queue) {
    fillBuf(values, queue);
  }
  Tensor3(const std::vector<std::vector<std::vector<float>>> &values,
          const cl::CommandQueue *queue = nullptr)
      : Tensor({(int)values.size(), (int)values[0].size(),
                (int)values[0][0].size()},
               false, queue) {
    std::vector<float> v(shape[0] * shape[1] * shape[2]);
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j)
        for (int k = 0; k < shape[2]; ++k)
          v[i * shape[1] * shape[2] + j * shape[1] + k] = values[i][j][k];
    }
    fillBuf(v, queue);
  }
  Tensor3(const Tensor3 &other, const cl::CommandQueue *queue = nullptr)
      : Tensor(other, queue) {};
  Tensor3 &operator=(const Tensor3 &other) {
    Tensor::operator=(other);
    return *this;
  };
  Tensor3(Tensor3 &&other) : Tensor(std::move(other)) {};
  Tensor3 &operator=(Tensor3 &&other) {
    Tensor::operator=(std::move(other));
    return *this;
  };
};

typedef Tensor0 Scalar;
typedef Tensor1 Vector;
typedef Tensor2 Matrix;

} // namespace GPU
