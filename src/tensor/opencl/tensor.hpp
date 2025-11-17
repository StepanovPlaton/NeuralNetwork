#pragma once

#include "opencl.hpp"

#include "../tensor.hpp"

template <typename T, int Dim> class Tensor : public ITensor<T, Dim> {
private:
  cl::Buffer *data_ = nullptr;
  cl::Event event_ = cl::Event();

  template <typename... Events> std::vector<cl::Event> all(Events &&...events) {
    return {std::forward<Events>(events)...};
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
                                         all(event_), &event_);
  }
  void fillBuf(size_t size, cl::Buffer *data) {
    createBuf(size);
    openCL.getQueue().enqueueWriteBuffer(*data_, CL_FALSE, 0,
                                         data.size() * sizeof(T), other..data(),
                                         all(event_), &event_);
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
  Tensor(const std::array<size_t, Dim> &shape, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<T> data(getSize());
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> dis(min, max);
      for (T &e : data_)
        e = dis(gen);
    } else if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min, max);
      for (T &e : data_)
        e = dis(gen);
    } else
      throw std::invalid_argument("Invalid randomized type");
    fillBuf(data);
  }

  Tensor(const Tensor &other) : ITensor(other.shape) {
    createBuf(other.getSize());
    q.enqueueCopyBuffer(*other.buffer, *buffer, 0, 0,
                        other.getSize() * sizeof(float));
  }
  Tensor &operator=(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;
  ~Tensor() = default;

  T &operator[](size_t i);
  const T &operator[](size_t i) const;
  template <typename... Indices> T &operator()(Indices... indices);
  template <typename... Indices> const T &operator()(Indices... indices) const;

  using ITensor::operator+;
  using ITensor::operator-;

  Tensor operator+() const override;
  Tensor operator-() const override;

  Tensor &operator+=(const T &scalar) override;

  Tensor &operator*=(const T &scalar) override;

  Tensor &operator+=(const Tensor &other) override;

  Tensor &operator*=(const Tensor &other) override;

  Tensor<T, Dim == 1 ? 0 : 2> operator%(const Tensor &other) const;

  std::string toString() const override;
};

#include "tensor.tpp"

#include "../fabric.hpp"