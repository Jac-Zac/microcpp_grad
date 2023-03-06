#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <sstream>

#include <micrograd/engine.hpp>
#include <micrograd/nn.hpp>

namespace py = pybind11;
using namespace nn;

PYBIND11_MODULE(pymicrograd, handle) {
  py::class_<Value<double>>(handle, "Value")
      .def(py::init<double,std::string>())
      /* .def_property("data", Value<double>::data) */
      /* .def_property("grad", Value<double>::grad) */
      /* .def_property("label", this->label;) */
      .def("backward", [](Value<double>& val) { val.backward(); })
      .def("draw_graph", [](Value<double>& val) { val.draw_graph(); })
      .def("tanh", [](Value<double>& val) { return val.tanh(); })
      .def("__add__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs + rhs; })
      .def("__radd__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs + rhs; })
      .def("__sub__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs - rhs; }) .def("__rsub__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs - rhs; })
      .def("__mul__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs * rhs; })
      .def("__rmul__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs * rhs; })
      .def("__truediv__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs / rhs; })
      .def("__pow__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs ^ rhs; })
      .def("__pow__", [](const Value<double>& lhs,const Value<double>& rhs) { return lhs ^ rhs; })
      .def("__repr__", [](const Value<double>& val) {
        std::stringstream ss;
        ss << val;
        return ss.str();
      });

  py::class_<Module<double>>(handle, "Module")
    .def(py::init<>())
    .def("zero_grad", &Module<double>::zero_grad)
    .def("parameters", &Module<double>::parameters);

  py::class_<Neuron<double>, Module<double>>(handle, "Neuron")
    .def(py::init<size_t>())
    .def("__call__", &Neuron<double>::operator())
    .def_property_readonly("parameters", &Neuron<double>::parameters);

  py::class_<Layer<double>, Module<double>>(handle, "Layer")
    .def(py::init<size_t, size_t>())
    .def("__call__", &Layer<double>::operator())
    .def_property_readonly("parameters", &Layer<double>::parameters);

  py::class_<MLP<double,3>, Module<double>>(handle, "MLP")
    .def(py::init<size_t, std::array<size_t,3>>())
    .def("__call__", &MLP<double,3>::operator())
    .def("parameters", &MLP<double,3>::parameters)
    .def("__repr__", [](const MLP<double,3>& mlp) {
        std::stringstream ss;
        ss << mlp;
        return ss.str();
    });
}
