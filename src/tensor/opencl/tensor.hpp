#pragma once

#include "opencl.hpp"

#include "../tensor.hpp"

#include <iostream>
#include <random>
#include <sstream>
template <typename T, int Dim> class Tensor : public ITensor<T, Dim> {
private:
  cl::Buffer *data_ = nullptr;
  cl::Event event_ = cl::Event();

  class AutoEventList {
  private:
    std::vector<cl::Event> events_;

  public:
    AutoEventList(std::initializer_list<cl::Event> events) : events_(events) {}
    operator const std::vector<cl::Event> *() const { return &events_; }
  };
  template <typename... Events> AutoEventList all(Events &&...events) const {
    return AutoEventList{std::forward<Events>(events)...};
  }

  void createBuf(size_t size) {
    if (data_ != nullptr)
      throw std::runtime_error("Tensor buffer already exists");
    data_ = new cl::Buffer(openCL.getContext(), CL_MEM_READ_WRITE,
                           size * sizeof(T));
  }

  void fillBuf(const std::vector<T> &data) {
    createBuf(data.size());
    openCL.getQueue().enqueueWriteBuffer(*data_, CL_FALSE, 0,
                                         data.size() * sizeof(T), data.data(),
                                         nullptr, &event_);
  }
  void fillBuf(const Tensor &other) {
    createBuf(other.getSize());
    openCL.getQueue().enqueueCopyBuffer(*other.getData(), *data_, 0, 0,
                                        other.getSize() * sizeof(T),
                                        all(other.getEvent()), &event_);
  }

public:
  typedef class ITensor<T, Dim> ITensor;

  using ITensor::axes_;
  using ITensor::checkAxisInDim;
  using ITensor::checkItHasSameShape;
  using ITensor::computeIndex;
  using ITensor::getSize;
  using ITensor::shape_;

  Tensor() = delete;
  Tensor(const std::array<size_t, Dim> &shape) : ITensor(shape) {
    createBuf(getSize());
  };
  Tensor(const std::array<size_t, Dim> &shape, T value) : ITensor(shape) {
    std::vector<T> data(getSize());
    std::fill(data.begin(), data.end(), value);
    fillBuf(data);
  }
  Tensor(const std::array<size_t, Dim> &shape, const std::vector<T> &data)
      : ITensor(shape) {
    fillBuf(data);
  }
  Tensor(const std::array<size_t, Dim> &shape, T min, T max) : ITensor(shape) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<T> data(getSize());
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> dis(min, max);
      for (T &e : data)
        e = dis(gen);
    } else if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min, max);
      for (T &e : data)
        e = dis(gen);
    } else
      throw std::invalid_argument("Invalid randomized type");
    fillBuf(data);
  }

  Tensor(const Tensor &other) : ITensor(other) {
    event_ = other.event_;
    fillBuf(other);
  }
  Tensor &operator=(const Tensor &other) {
    ITensor::operator=(other);
    event_ = other.event_;
    fillBuf(other);
    return *this;
  }
  Tensor(Tensor &&other) noexcept : ITensor(std::move(other)) {
    data_ = other.data_;
    event_ = other.event_;
    other.data_ = nullptr;
  }
  Tensor &operator=(Tensor &&other) noexcept {
    ITensor::operator=(std::move(other));
    data_ = other.data_;
    event_ = other.event_;
    other.data = nullptr;
    return *this;
  }
  ~Tensor() {
    if (data_ != nullptr)
      delete data_;
  };

  const cl::Buffer *getData() const { return data_; }
  const cl::Event &getEvent() const { return event_; }

  // T &operator[](size_t i);
  // const T &operator[](size_t i) const;
  // template <typename... Indices> T &operator()(Indices... indices);
  // template <typename... Indices> const T &operator()(Indices... indices)
  // const;

  using ITensor::operator+;
  using ITensor::operator-;

  Tensor operator+() override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::POSITIVE);
    kernel.setArg(0, *data_);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor operator-() override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::NEGATIVE);
    kernel.setArg(0, *data_);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor &operator+=(const T scalar) override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::S_ADD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, scalar);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor &operator*=(const T scalar) override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::S_MULT);
    kernel.setArg(0, *data_);
    kernel.setArg(1, scalar);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor &operator+=(const Tensor &other) override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::T_ADD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, *other.getData());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(getSize()), cl::NullRange,
        all(event_, other.event_), &event_);
    return *this;
  }

  Tensor &operator*=(const Tensor &other) override {
    cl::Kernel kernel = openCL.createKernel(OpenCL::Method::T_HADAMARD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, *other.getData());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(getSize()), cl::NullRange,
        all(event_, other.event_), &event_);
    return *this;
  }

#define TILE_SIZE 16
  Tensor<T, Dim == 1 ? 0 : 2> operator%(const Tensor &other) const {
    static_assert(Dim == 1 || Dim == 2,
                  "Inner product is only defined for vectors and matrices");
    if constexpr (Dim == 1) {
      static_assert(false, "TODO vector scalar multiplication");
    } else if constexpr (Dim == 2) {
      if (shape_[axes_[1]] != other.shape_[other.axes_[0]])
        throw std::invalid_argument(
            "Matrix dimensions must match for multiplication");
      int m = (int)shape_[axes_[0]];
      int k = (int)shape_[axes_[1]];
      int n = (int)other.shape_[other.axes_[1]];
      Tensor<T, 2> result({m, n});
      cl::Kernel kernel = openCL.createKernel(OpenCL::Method::T_MULT);
      kernel.setArg(0, *data_);
      kernel.setArg(1, *other.getData());
      kernel.setArg(2, *result.getData());
      kernel.setArg(3, m);
      kernel.setArg(4, n);
      kernel.setArg(5, k);
      cl::NDRange global_size(((m + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
                              ((n + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE);
      cl::NDRange local_size(TILE_SIZE, TILE_SIZE);
      openCL.getQueue().enqueueNDRangeKernel(
          kernel, cl::NullRange, global_size, local_size,
          all(event_, other.event_), &result.event_);
      return result;
    }
  }

  std::string toString() const override {
    std::vector<float> result(getSize());
    openCL.getQueue().enqueueReadBuffer(
        *data_, CL_TRUE, 0, getSize() * sizeof(T), result.data(), all(event_));
    std::ostringstream oss;
    if constexpr (Dim == 0) {
      oss << "Scalar<" << typeid(T).name() << ">: " << result[0];
    } else if constexpr (Dim == 1) {
      oss << "Vector<" << typeid(T).name() << ">(" << shape_[0] << "): [";
      for (size_t i = 0; i < getSize(); ++i) {
        oss << result[i];
        if (i < getSize() - 1)
          oss << ", ";
      }
      oss << "]";
    } else if constexpr (Dim == 2) {
      oss << "Matrix<" << typeid(T).name() << ">(" << shape_[axes_[0]] << "x"
          << shape_[axes_[1]] << "):";
      for (size_t i = 0; i < shape_[axes_[0]]; ++i) {
        oss << "\n  [";
        for (size_t j = 0; j < shape_[axes_[1]]; ++j) {
          oss << result[i * shape_[axes_[0]] + j];
          if (j < shape_[axes_[1]] - 1)
            oss << ", ";
        }
        oss << "]";
      }
    } else {
      oss << "Tensor" << Dim << "D<" << typeid(T).name() << ">" << "[";
      for (size_t i = 0; i < Dim; ++i) {
        oss << shape_[axes_[i]];
        if (i < Dim - 1)
          oss << "x";
      }
      oss << "]: [";
      size_t show = std::min(getSize(), size_t(10));
      for (size_t i = 0; i < show; ++i) {
        oss << result[i];
        if (i < show - 1)
          oss << ", ";
      }
      if (getSize() > 10)
        oss << ", ...";
      oss << "]";
    }
    return oss.str();
  }
};

#include "tensor.tpp"

#include "../fabric.hpp"
