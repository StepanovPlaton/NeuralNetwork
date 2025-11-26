#pragma once

#include "opencl.hpp"

#include "kernels.hpp"

#include "../tensor.hpp"

#include <random>

template <typename T, int Dim> class Tensor : public ITensor<T, Dim> {
private:
  cl::Buffer *data_ = nullptr;
  mutable cl::Event event_ = cl::Event();

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

  constexpr const static Kernels<T>::Vector vector = Kernels<T>::Vector::type1;
  constexpr const static int vectorSize = (int)vector;
  constexpr const static int tileSize = vectorSize * 4;

  static cl::Kernel createKernel(Kernels<T>::Method method) {
    static Kernels<T> kernels(vector);
    return kernels.create(method);
  }

public:
  typedef class ITensor<T, Dim> ITensor;

  using ITensor::axes_;
  using ITensor::checkAxisInDim;
  using ITensor::checkItHasSameShape;
  // using ITensor::computeIndex;
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
    other.data_ = nullptr;
    return *this;
  }
  ~Tensor() {
    if (data_ != nullptr)
      delete data_;
  };

  const cl::Buffer *getData() const { return data_; }
  const cl::Event &getEvent() const { return event_; }

  using ITensor::operator+;
  using ITensor::operator-;

  Tensor operator+() const override {
    Tensor result = *this;
    cl::Kernel kernel = createKernel(Kernels<T>::Method::POSITIVE);
    kernel.setArg(0, *result.getData());
    kernel.setArg(1, (int)result.getSize());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(result.getSize()), cl::NullRange,
        all(result.event_), &result.event_);
    return result;
  }

  Tensor operator-() const override {
    Tensor result = *this;
    cl::Kernel kernel = createKernel(Kernels<T>::Method::NEGATIVE);
    kernel.setArg(0, *result.getData());
    kernel.setArg(1, (int)result.getSize());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(result.getSize()), cl::NullRange,
        all(result.event_), &result.event_);
    return result;
  }

  Tensor &operator+=(const T scalar) override {
    cl::Kernel kernel = createKernel(Kernels<T>::Method::S_ADD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, (int)getSize());
    kernel.setArg(2, scalar);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor &operator*=(const T scalar) override {
    cl::Kernel kernel = createKernel(Kernels<T>::Method::S_MULT);
    kernel.setArg(0, *data_);
    kernel.setArg(1, (int)getSize());
    kernel.setArg(2, scalar);
    openCL.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(getSize()),
                                           cl::NullRange, all(event_), &event_);
    return *this;
  }

  Tensor &operator+=(const Tensor &other) override {
    checkItHasSameShape(other);
    cl::Kernel kernel = createKernel(Kernels<T>::Method::T_ADD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, *other.getData());
    kernel.setArg(2, (int)getSize());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(getSize()), cl::NullRange,
        all(event_, other.event_), &event_);
    return *this;
  }

  Tensor &operator*=(const Tensor &other) override {
    checkItHasSameShape(other);
    cl::Kernel kernel = createKernel(Kernels<T>::Method::T_HADAMARD);
    kernel.setArg(0, *data_);
    kernel.setArg(1, *other.getData());
    kernel.setArg(2, (int)getSize());
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(getSize()), cl::NullRange,
        all(event_, other.event_), &event_);
    return *this;
  }

  Tensor<T, Dim == 1 ? 0 : 2> operator%(const Tensor &other) const {
    static_assert(Dim == 1 || Dim == 2,
                  "Inner product is only defined for vectors and matrices");
    if constexpr (Dim == 1) {
      static_assert(false, "TODO vector scalar multiplication");
    } else if constexpr (Dim == 2) {
      if (shape_[axes_[1]] != other.shape_[other.axes_[0]])
        throw std::invalid_argument(
            "Matrix dimensions must match for multiplication");
      size_t m = shape_[axes_[0]];
      size_t k = shape_[axes_[1]];
      size_t n = other.shape_[other.axes_[1]];
      Tensor<T, 2> result({m, n});
      cl::Kernel kernel = createKernel(Kernels<T>::Method::T_MULT);
      kernel.setArg(0, *data_);
      kernel.setArg(1, *other.getData());
      kernel.setArg(2, *result.getData());
      kernel.setArg(3, (int)m);
      kernel.setArg(4, (int)n);
      kernel.setArg(5, (int)k);
      cl::NDRange globalSize(m, n);
      openCL.getQueue().enqueueNDRangeKernel(
          kernel, cl::NullRange, globalSize, cl::NullRange,
          all(event_, other.event_), &result.event_);
      return result;
    }
  }

  Tensor apply(Function f, bool derivative = false) const override {
    Tensor result = *this;
    cl::Kernel kernel = createKernel(Kernels<T>::Method::FUNC);
    kernel.setArg(0, *result.getData());
    kernel.setArg(1, (int)f);
    kernel.setArg(2, (int)derivative);
    openCL.getQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(result.getSize()), cl::NullRange,
        all(result.event_), &result.event_);
    return result;
  };

  std::string toString() const override {
    std::vector<T> result(getSize());
    openCL.getQueue().enqueueReadBuffer(*data_, CL_FALSE, 0,
                                        getSize() * sizeof(T), result.data(),
                                        all(event_), &event_);
    event_.wait();
    return ITensor::format(result);
  }
};

#include "tensor.tpp"

#include "../fabric.hpp"
