#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

#include <ATen/Parallel.h>
#include <omp.h>
#include <iostream>
#include <c10/util/irange.h>


namespace test {

void foreach_non_finite_check_and_unscale(std::vector<at::Tensor> tensors, at::Tensor &found_inf, const at::Tensor &inv_scale)
{

    auto* found_inf_ptr = found_inf.data_ptr<float>();
    auto* inv_scale_ptr = inv_scale.data_ptr<float>();
    
    const auto inv_scale_val = *inv_scale_ptr;
    
    int max_threads = tensors.size() > 10 ? 10 : tensors.size();

    // std::cout << " omp_get_thread_num() " << omp_get_thread_num() << std::endl;

    if (inv_scale_val == 1.f) {
      // #ifdef _OPENMP
      // #pragma omp parallel for num_threads(22)
      // #endif
      for (int i=0; i<tensors.size(); i++) {
          if (at::native::isfinite(tensors[i]).all().item<bool>()) {
              *found_inf_ptr = 1.f;
          }    
      }
    } else {
      #ifdef _OPENMP
      // #pragma omp parallel for schedule(simd:dynamic)
      #pragma omp parallel for num_threads(2)
      #endif
      for (int i=0; i<tensors.size(); i++) {
          // // #ifdef _OPENMP
          // // #pragma omp parallel for num_threads(22)
          // // #endif
          // for (const auto j : c10::irange(tensors[i].numel())) {
          //   auto isfinite = at::native::isfinite(tensors[i][j]).all().item<bool>();
          //   tensors[i][j] *= inv_scale_val;
          //   if (!isfinite) {
          //     *found_inf_ptr = 1.f;
          //     // break;
          //   }
          // }
          if (at::native::isfinite(tensors[i]).all().item<bool>()) {
              *found_inf_ptr = 1.f;
          }
          tensors[i].mul_(inv_scale_val); 
      }
    }
}

} // end of namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("foreach_non_finite_check_and_unscale", 
        &test::foreach_non_finite_check_and_unscale,
	"CPU version of at::_amp_foreach_non_finite_check_and_unscale_.",
  py::call_guard<py::gil_scoped_release>());
}