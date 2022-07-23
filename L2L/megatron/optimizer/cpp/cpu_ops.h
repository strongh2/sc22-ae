#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>
#include <Python.h>

bool optimizer::has_attr(
    at::Tensor &obj,
    std::string const &attrName);
int optimizer::set_attr_to_none(
    at::Tensor &obj,
    std::string const &attrName);
int optimizer::set_attr_from_this_to_that(
    at::Tensor &_that,
    std::string const &_thatAttrName,
    at::Tensor &_this,
    std::string const &_thisAttrName);
bool optimizer::attr_is_none(
    at::Tensor &obj,
    std::string const &attrName);


void optimizer::_copy_model_grads_to_main_grads(
    std::vector<std::vector<at::Tensor>> float16_groups,
    std::vector<std::vector<at::Tensor>> fp32_from_float16_groups,
    std::vector<std::vector<at::Tensor>> fp32_from_fp32_groups,
    const bool params_have_main_grad,
    const bool use_contiguous_buffers_in_local_ddp);

void optimizer::foreach_non_finite_check_and_unscale(
    std::vector<at::Tensor> tensors,
    at::Tensor &found_inf,
    const at::Tensor &inv_scale);
void optimizer::_multi_tensor_copy_this_to_that(
    std::vector<at::Tensor> these,
    std::vector<at::Tensor> those);