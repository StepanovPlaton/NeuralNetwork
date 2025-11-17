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

                    .def("get_shape", &TensorInfo<T, Dim>::getShape)
                    .def("get_axes", &TensorInfo<T, Dim>::getAxes)
                    .def("get_size", &TensorInfo<T, Dim>::getSize)

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
                    .def(py::self += T())
                    .def(py::self -= T())
                    .def(py::self *= T())
                    .def(py::self /= T())
                    .def(T() + py::self)
                    .def(T() - py::self)
                    .def(T() * py::self)

                    .def("__pos__", [](const Tensor<T, Dim> &t) { return +t; })
                    .def("__neg__", [](const Tensor<T, Dim> &t) { return -t; })

                    .def("__repr__", &Tensor<T, Dim>::toString);

  if constexpr (Dim >= 2) {
    tensor
        .def("transpose", py::overload_cast<const std::array<int, Dim> &>(
                              &Tensor<T, Dim>::transpose))
        .def("transpose",
             py::overload_cast<int, int>(&Tensor<T, Dim>::transpose))
        .def("t", &Tensor<T, Dim>::t);
  }

  if constexpr (Dim != 0)
    tensor
        .def(
            "__getitem__",
            [](Tensor<T, Dim> &t, size_t index) -> T & {
              if (index >= t.getSize())
                throw py::value_error("Index out of range");
              return t[index];
            },
            py::return_value_policy::reference)
        .def(
            "__getitem__",
            [](Tensor<T, Dim> &t, const py::tuple &indices) -> T & {
              if (indices.size() != Dim)
                throw py::value_error("Expected " + std::to_string(Dim) +
                                      " indices, got " +
                                      std::to_string(indices.size()));
              return [&]<size_t... I>(std::index_sequence<I...>) -> T & {
                return t(py::cast<size_t>(indices[I])...);
              }(std::make_index_sequence<Dim>{});
            },
            py::return_value_policy::reference)

        .def("__setitem__",
             [](Tensor<T, Dim> &t, size_t index, const T &value) {
               if (index >= t.getSize())
                 throw py::value_error("Index out of range");
               t[index] = value;
             })
        .def("__setitem__",
             [](Tensor<T, Dim> &t, const py::tuple &indices, const T &value) {
               if (indices.size() != Dim)
                 throw py::value_error("Expected " + std::to_string(Dim) +
                                       " indices, got " +
                                       std::to_string(indices.size()));
               [&]<size_t... I>(std::index_sequence<I...>) {
                 t(py::cast<size_t>(indices[I])...) = value;
               }(std::make_index_sequence<Dim>{});
             });

  if constexpr (Dim == 1 || Dim == 2)
    tensor.def("__matmul__", &Tensor<T, Dim>::operator%);
}

PYBIND11_MODULE(tensor, m) {
  m.doc() = "Tensor math library";

  register_tensor<float, 0>(m, "Scalar");
  register_tensor<float, 1>(m, "Vector");
  register_tensor<float, 2>(m, "Matrix");
  // register_tensor<float, 3>(m, "Tensor3");
  //
  // register_tensor<int, 0>(m, "iScalar");
  // register_tensor<int, 1>(m, "iVector");
  // register_tensor<int, 2>(m, "iMatrix");
  // register_tensor<int, 3>(m, "iTensor3");
}
