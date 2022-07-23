#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

void test::foreach_non_finite_check_and_unscale(std::vector<at::Tensor> tensors, at::Tensor &found_inf, const at::Tensor &inv_scale);
