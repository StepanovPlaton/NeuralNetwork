#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"

namespace py = pybind11;

template <typename T, int Dim>
void register_tensor(py::module &m, const std::string &name) {
  auto tensor = py::class_<Tensor<T, Dim>>(m, name.c_str())
                    .def(py::init<const std::array<size_t, Dim> &>())
                    .def(py::init<const std::array<size_t, Dim> &, T>())
                    .def(py::init<const std::array<size_t, Dim> &,
                                  const std::vector<T> &>())
                    .def(py::init<const std::array<size_t, Dim> &, T, T>())

                    .def("get_shape", &Tensor<T, Dim>::getShape)
                    .def("get_data", &Tensor<T, Dim>::getData)
                    .def("get_size", &Tensor<T, Dim>::getSize)
                    .def("get_axes", &Tensor<T, Dim>::getAxes)

                    .def("__getitem__",
                         [](const Tensor<T, Dim> &t, size_t i) -> T {
                           if (i >= t.getSize())
                             throw py::index_error();
                           return t[i];
                         })
                    .def("__setitem__",
                         [](Tensor<T, Dim> &t, size_t i, T value) {
                           if (i >= t.getSize())
                             throw py::index_error();
                           t[i] = value;
                         })

                    // .def("__call__",
                    //      [](Tensor<T, Dim> &t, py::args args) -> T & {
                    //
                    //      })

                    .def(py::self + py::self)
                    .def(py::self - py::self)
                    .def(py::self * py::self)
                    .def(py::self += py::self)
                    .def(py::self -= py::self)
                    .def(py::self *= py::self)

                    .def(py::self + T())
                    .def(py::self - T())
                    .def(py::self * T())
                    .def(py::self / T())
                    .def(T() + py::self)
                    .def(T() - py::self)
                    .def(T() * py::self)

                    .def(py::self += T())
                    .def(py::self -= T())
                    .def(py::self *= T())
                    .def(py::self /= T())

                    .def("__pos__", [](const Tensor<T, Dim> &t) { return +t; })
                    .def("__neg__", [](const Tensor<T, Dim> &t) { return -t; })

                    .def("print", &Tensor<T, Dim>::print);

  if constexpr (Dim == 1 || Dim == 2)
    tensor.def("__matmul__", &Tensor<T, Dim>::operator%);

  if constexpr (Dim >= 2) {
    tensor
        .def("transpose", py::overload_cast<const std::array<int, Dim> &>(
                              &Tensor<T, Dim>::transpose))
        .def("transpose",
             py::overload_cast<int, int>(&Tensor<T, Dim>::transpose))
        .def("t", &Tensor<T, Dim>::t);
  }
}

PYBIND11_MODULE(tensor, m) {
  m.doc() = "Tensor math library";

  register_tensor<float, 0>(m, "Scalar");
  register_tensor<float, 1>(m, "Vector");
  register_tensor<float, 2>(m, "Matrix");
  register_tensor<float, 3>(m, "Tensor3");
  register_tensor<float, 4>(m, "Tensor4");
  register_tensor<float, 5>(m, "Tensor5");

  register_tensor<double, 0>(m, "dScalar");
  register_tensor<double, 1>(m, "dVector");
  register_tensor<double, 2>(m, "dMatrix");
  register_tensor<double, 3>(m, "dTensor3");
  register_tensor<double, 4>(m, "dTensor4");
  register_tensor<double, 5>(m, "dTensor5");

  register_tensor<int, 0>(m, "iScalar");
  register_tensor<int, 1>(m, "iVector");
  register_tensor<int, 2>(m, "iMatrix");
  register_tensor<int, 3>(m, "iTensor3");
  register_tensor<int, 4>(m, "iTensor4");
  register_tensor<int, 5>(m, "iTensor5");
}