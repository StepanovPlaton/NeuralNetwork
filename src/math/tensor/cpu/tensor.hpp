#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "../tensor.hpp"

extern std::mt19937 gen;

namespace CPU {
class Tensor;
class Tensor0;
class Tensor1;
class Tensor2;
class Tensor3;

class Tensor : public ITensor {
protected:
  std::vector<float> data;

  void resize(size_t size) { data.resize(size); }
  void resize(const std::vector<int> &shape) {
    size_t size = 1;
    for (int dim : shape)
      size *= dim;
    resize(size);
  }

public:
  Tensor(const std::vector<int> &shape) : ITensor(shape) {
    resize(shape);
    std::generate(data.begin(), data.end(),
                  []() { return std::generate_canonical<float, 10>(gen); });
  }
  Tensor(const std::vector<int> &shape, float value) : ITensor(shape) {
    resize(shape);
    std::fill(data.begin(), data.end(), value);
  }
  Tensor(const std::vector<int> &shape, bool fill) : ITensor(shape) {
    resize(shape);
    if (fill)
      std::fill(data.begin(), data.end(), 0.0f);
  }
  Tensor(const Tensor &) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&other) = default;
  Tensor &operator=(Tensor &&other) = default;

  float &operator[](int index) { return data[index]; }
  const float &operator[](int index) const { return data[index]; }

  virtual void print() const {
    std::cout << "Tensor(" << getDim() << "): [";
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << data[i];
      if (i > 15) {
        std::cout << "... ";
        break;
      }
      if (i != data.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  std::vector<float> toVector() const { return data; }

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
  Tensor0(const std::vector<int> &shape) : Tensor(shape) {
    if (shape.size() != 0)
      throw std::invalid_argument("Tensor0 dimension must be 0");
  }
  Tensor0(const std::vector<int> &shape, float value) : Tensor(shape, value) {
    if (shape.size() != 0)
      throw std::invalid_argument("Tensor0 dimension must be 0");
  }
  Tensor0() : Tensor({}) {
    resize(1);
    data[0] = std::generate_canonical<float, 10>(gen);
  }
  Tensor0(float value) : Tensor({}) {
    resize(1);
    data[0] = value;
  }
  Tensor0(const Tensor0 &) = default;
  Tensor0 &operator=(const Tensor0 &) = default;
  Tensor0(Tensor0 &&other) = default;
  Tensor0 &operator=(Tensor0 &&other) = default;

  void print() const override {
    std::cout << "Scalar: " << data[0] << std::endl;
  }

  float &value() { return data[0]; }
  const float &value() const { return data[0]; }
};

class Tensor1 : public Tensor, public ITensor1 {
public:
  Tensor1(const std::vector<int> &shape) : Tensor(shape) {
    if (shape.size() != 1)
      throw std::invalid_argument("Tensor1 dimension must be 1");
  }
  Tensor1(const std::vector<int> &shape, float value) : Tensor(shape, value) {
    if (shape.size() != 1)
      throw std::invalid_argument("Tensor1 dimension must be 1");
  }
  Tensor1(int size) : Tensor({size}) {}
  Tensor1(int size, float value) : Tensor({size}, value) {}
  Tensor1(const std::vector<float> &values) : Tensor({(int)values.size()}) {
    data = values;
  }
  Tensor1(const Tensor1 &) = default;
  Tensor1 &operator=(const Tensor1 &) = default;
  Tensor1(Tensor1 &&other) = default;
  Tensor1 &operator=(Tensor1 &&other) = default;

  void print() const override {
    std::cout << "Vector(" << shape[0] << "): [";
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << data[i];
      if (i != data.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  float &operator()(int i) { return data[i]; }
  const float &operator()(int i) const { return data[i]; }

  int getSize() const override { return shape[0]; }
};

class Tensor2 : public ITensor2, public Tensor {
public:
  Tensor2(const std::vector<int> &shape) : Tensor(shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("Tensor2 dimension must be 2");
  }
  Tensor2(const std::vector<int> &shape, float value) : Tensor(shape, value) {
    if (shape.size() != 2)
      throw std::invalid_argument("Tensor2 dimension must be 2");
  }
  Tensor2(int rows, int cols) : ITensor2(), Tensor({rows, cols}) {}
  Tensor2(int rows, int cols, float value)
      : ITensor2(), Tensor({rows, cols}, value) {}
  Tensor2(int rows, int cols, const std::vector<float> &values)
      : Tensor({rows, cols}, false) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        data[i * shape[1] + j] = values[i * shape[1] + j];
      }
    }
  }
  Tensor2(const std::vector<std::vector<float>> &values)
      : Tensor({(int)values.size(), (int)values[0].size()}) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        data[i * shape[1] + j] = values[i][j];
      }
    }
  }
  Tensor2(const Tensor2 &) = default;
  Tensor2 &operator=(const Tensor2 &) = default;
  Tensor2(Tensor2 &&other) = default;
  Tensor2 &operator=(Tensor2 &&other) = default;

  void print() const override {
    std::cout << "Matrix(" << shape[0] << "x" << shape[1] << "):\n";
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        std::cout << data[i * shape[1] + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  float &operator()(int i, int j) { return data[i * shape[1] + j]; }
  const float &operator()(int i, int j) const { return data[i * shape[1] + j]; }

  int getRows() const override { return shape[0]; }
  int getCols() const override { return shape[1]; }
};

class Tensor3 : public Tensor, public ITensor3 {
public:
  Tensor3(const std::vector<int> &shape) : Tensor(shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("Tensor3 dimension must be 3");
  }
  Tensor3(const std::vector<int> &shape, float value) : Tensor(shape, value) {
    if (shape.size() != 3)
      throw std::invalid_argument("Tensor3 dimension must be 3");
  }
  Tensor3(int d1, int d2, int d3) : Tensor({d1, d2, d3}) {}
  Tensor3(int d1, int d2, int d3, float value) : Tensor({d1, d2, d3}, value) {}
  Tensor3(int d1, int d2, int d3, const std::vector<float> &values)
      : Tensor({d1, d2, d3}, false) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        for (int k = 0; k < shape[2]; ++k) {
          data[i * shape[1] * shape[2] + j * shape[2] + k] =
              values[i * shape[1] * shape[2] + j * shape[2] + k];
        }
      }
    }
  }
  Tensor3(const std::vector<std::vector<std::vector<float>>> &values)
      : Tensor({(int)values.size(), (int)values[0].size(),
                (int)values[0][0].size()}) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        for (int k = 0; k < shape[2]; ++k) {
          data[i * shape[1] * shape[2] + j * shape[2] + k] = values[i][j][k];
        }
      }
    }
  }
  Tensor3(const Tensor3 &) = default;
  Tensor3 &operator=(const Tensor3 &) = default;
  Tensor3(Tensor3 &&other) = default;
  Tensor3 &operator=(Tensor3 &&other) = default;

  void print() const override {
    std::cout << "Tensor3(" << shape[0] << "x" << shape[1] << "x" << shape[2]
              << "):\n";
    for (int i = 0; i < shape[0]; ++i) {
      std::cout << "Slice " << i << ":\n";
      for (int j = 0; j < shape[1]; ++j) {
        for (int k = 0; k < shape[2]; ++k) {
          std::cout << data[i * shape[1] * shape[2] + j * shape[2] + k] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  float &operator()(int i, int j, int k) {
    return data[i * shape[1] * shape[2] + j * shape[2] + k];
  }
  const float &operator()(int i, int j, int k) const {
    return data[i * shape[1] * shape[2] + j * shape[2] + k];
  }
};

typedef Tensor0 Scalar;
typedef Tensor1 Vector;
typedef Tensor2 Matrix;

} // namespace CPU
