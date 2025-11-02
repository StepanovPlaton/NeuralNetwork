#pragma once

#include <random>
#include <stdexcept>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());

class ITensor {
protected:
  std::vector<int> shape;

  void validateDimensions(const std::vector<int> &shape) const {
    if (shape.empty())
      throw std::invalid_argument("Tensor shape cannot be empty");
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0)
        throw std::invalid_argument(
            "All tensor dimensions must be positive, but dimension " +
            std::to_string(i) + " is " + std::to_string(shape[i]));
    }
  };

public:
  ITensor(const std::vector<int> &shape) : shape(shape) {}
  ITensor(const ITensor &other) : shape(other.shape) {};
  ITensor &operator=(const ITensor &other) {
    shape = other.shape;
    return *this;
  };
  ITensor(ITensor &&other) : shape(other.shape) {};
  ITensor &operator=(ITensor &&other) {
    shape = other.shape;
    return *this;
  };

  const std::vector<int> &getShape() const { return shape; }
  int getDim() const { return static_cast<int>(shape.size()); }
  size_t getSize() const {
    size_t size = 1;
    for (int dim : shape)
      size *= dim;
    return size;
  };

  enum class Type { SCALAR, VECTOR, MATRIX, TENSOR3 };
  Type getType() const { return static_cast<Type>(shape.size()); };
};

class ITensor0 {};

class ITensor1 {};

class ITensor2 {
public:
  virtual int getRows() const = 0;
  virtual int getCols() const = 0;
};

class ITensor3 {};

typedef ITensor0 IScalar;
typedef ITensor1 IVector;
typedef ITensor2 IMatrix;
