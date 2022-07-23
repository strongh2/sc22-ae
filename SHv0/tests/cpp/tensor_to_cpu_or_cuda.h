#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

// void test::tensor_to_cpu(std::vector<at::Tensor> &tensors);
// void test::tensor_to_cuda(std::vector<at::Tensor> &tensors);

// void test::tensor_to_cpu(std::vector<std::pair<std::string, at::Tensor>> &tensors);
// void test::tensor_to_cuda(std::vector<std::pair<std::string, at::Tensor>> &tensors);

std::map<std::string, std::map<std::string, at::Tensor>> test::tensor_to_cpu(std::map<std::string, std::map<std::string, at::Tensor>> &tensors);
std::map<std::string, std::map<std::string, at::Tensor>> test::tensor_to_cuda(std::map<std::string,std::map<std::string, at::Tensor>> &tensors);