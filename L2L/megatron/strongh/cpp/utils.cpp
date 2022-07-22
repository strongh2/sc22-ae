#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

#include <omp.h>
#include <iostream>
#include <torch/csrc/autograd/python_variable.h>

#include <c10/util/irange.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>
// #include <torch/csrc/Export.h>

#include <utility>

// namespace torch {

// void save(
//     const std::vector<torch::Tensor>& tensor_vec,
//     std::string&&... args) {
//     torch::save(tensor_vec, args);
// }

// std::vector<torch::Tensor> load(
//     std::string&&... args) {
//     std::vector<torch::Tensor> tensor_vec,
//     torch::load(tensor_vec, args);
//     return tensor_vec;
// }

// } // namespace torch


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("save",
//         &save,
//         "torch::save",
//         py::call_guard<py::gil_scoped_release>());

//     m.def("load",
//         &load,
//         "torch::load",
//         py::call_guard<py::gil_scoped_release>());
// }