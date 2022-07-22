#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

#include <ATen/Parallel.h>
#include <omp.h>
#include <iostream>
#include <c10/util/irange.h>


namespace test {

std::map<std::string, std::map<std::string, at::Tensor>> tensor_to_cpu(std::map<std::string, std::map<std::string, at::Tensor>> &tensors)
{
    std::map<std::string, std::map<std::string, at::Tensor>> results;

    #pragma omp parallel for num_threads(8)
    for (int i=0; i<tensors.size(); i++) {
        auto itr = tensors.begin();
        std::advance(itr, i);
        
        std::map<std::string, at::Tensor> _res;

        for (auto const & t: itr->second) {
            _res.insert(std::pair<std::string, at::Tensor>(t.first, t.second.to(torch::Device(torch::kCPU), true)));
        }
        results.insert(std::pair<std::string, std::map<std::string, at::Tensor>>(itr->first, _res));
    }
    return results;
}

std::map<std::string, std::map<std::string, at::Tensor>> tensor_to_cuda(std::map<std::string,std::map<std::string, at::Tensor>> &tensors)
{   
    std::map<std::string, std::map<std::string, at::Tensor>> results;

    #pragma omp parallel for num_threads(8)
    for (int i=0; i<tensors.size(); i++) {
        auto itr = tensors.begin();
        std::advance(itr, i);
        
        std::map<std::string, at::Tensor> _res;

        for (auto const & t: itr->second) {
            _res.insert(std::pair<std::string, at::Tensor>(t.first, t.second.to({torch::kCUDA, 0}, true)));
        }
        results.insert(std::pair<std::string, std::map<std::string, at::Tensor>>(itr->first, _res));
    }
    return results;
}


} // end of namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_to_cpu", 
        &test::tensor_to_cpu,
	"tensor_to_cpu.",
  py::call_guard<py::gil_scoped_release>());

  m.def("tensor_to_cuda", 
        &test::tensor_to_cuda,
	"tensor_to_cuda.",
  py::call_guard<py::gil_scoped_release>());

}