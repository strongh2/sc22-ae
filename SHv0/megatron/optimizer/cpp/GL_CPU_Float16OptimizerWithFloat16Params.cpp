#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <math.h>

#include <omp.h>
#include <iostream>
#include <torch/csrc/autograd/python_variable.h>

// #ifdef _OPENMP
// omp_set_num_threads(10);
// #endif

// at::native::num_threads.store(10)

namespace optimizer {

void foreach_non_finite_check_and_unscale(std::vector<at::Tensor> tensors, at::Tensor &found_inf, const at::Tensor &inv_scale)
{
    auto* found_inf_ptr = found_inf.data_ptr<float>();
    auto* inv_scale_ptr = inv_scale.data_ptr<float>();
    
    const auto inv_scale_val = *inv_scale_ptr;
    
    int max_threads = tensors.size() > 10 ? 10 : tensors.size();

    // std::cout << " omp_get_thread_num() " << omp_get_thread_num() << std::endl;

    if (inv_scale_val == 1.f) {
        #ifdef _OPENMP
        //   , firstprivate(found_inf_ptr), lastprivate(found_inf_ptr)
        #pragma omp parallel for num_threads(4)     
        #endif
        for (int i=0; i<tensors.size(); i++) {
            if (at::native::isfinite(tensors[i]).all().item<bool>()) {
                *found_inf_ptr = 1.f;
            }    
        }
    } else {
        #ifdef _OPENMP
            //   , firstprivate(found_inf_ptr), lastprivate(found_inf_ptr)
        #pragma omp parallel for num_threads(4)
        #endif
        for (int i=0; i<tensors.size(); i++) {
            if (at::native::isfinite(tensors[i]).all().item<bool>()) {
                *found_inf_ptr = 1.f;
            }
            tensors[i].mul_(inv_scale_val);
      }
    }
}

bool has_attr(at::Tensor &obj, std::string const &attrName) {

    auto py_obj = THPVariable_Wrap(std::move(obj));

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    int result = PyObject_HasAttrString(
        py_obj,
        attrName.c_str());

    PyGILState_Release(gstate);

    return 1==result;
}

int set_attr_to_none(at::Tensor &obj, std::string const &attrName) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    int result = PyObject_SetAttrString(THPVariable_Wrap(std::move(obj)), attrName.c_str(), Py_None);

    PyGILState_Release(gstate);

    return result;
}

int set_attr_from_this_to_that(at::Tensor &_that, std::string const &_thatAttrName, 
                                at::Tensor &_this, std::string const &_thisAttrName) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    auto value = PyObject_GetAttrString(THPVariable_Wrap(std::move(_this)), _thisAttrName.c_str());
    int result = PyObject_SetAttrString(THPVariable_Wrap(std::move(_that)), _thatAttrName.c_str(), value);

    PyGILState_Release(gstate);
    return result;
}

bool attr_is_none(at::Tensor &obj, std::string const &attrName) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    auto result = PyObject_IsInstance(
        PyObject_GetAttrString(THPVariable_Wrap(std::move(obj)), attrName.c_str()), 
        Py_None
    );

    PyGILState_Release(gstate);
    return result;
}

void _copy_model_grads_to_main_grads(
    std::vector<std::vector<at::Tensor>> float16_groups, 
    std::vector<std::vector<at::Tensor>> fp32_from_float16_groups,
    std::vector<std::vector<at::Tensor>> fp32_from_fp32_groups,
    const bool params_have_main_grad,
    const bool use_contiguous_buffers_in_local_ddp)
{
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(4)
    #endif
    for (int i=0; i<float16_groups.size(); i++) {
        auto model_group = float16_groups[i];
        auto main_group = fp32_from_float16_groups[i];

        for (int j=0; j<model_group.size(); j++) {
            auto model_param = model_group[j];
            auto main_param = main_group[j];
    
            if (params_have_main_grad && has_attr(model_param, "main_grad")) {
                // main_param.grad = model_param.main_grad.float();
                ;
            } else {
                if (attr_is_none(model_param, "grad")) {
                    // main_param.grad = model_param.grad.float();
                    ;
                }
            }

            // model_param.grad = None
            set_attr_to_none(model_param, "grad");
        
            if (params_have_main_grad && !use_contiguous_buffers_in_local_ddp) {
                // model_param.main_grad = None
                set_attr_to_none(model_param, "main_grad");
            }
        }
    }

    if (params_have_main_grad) {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(4)
        #endif
        for (int i=0; i<fp32_from_fp32_groups.size(); i++) {
            auto model_group = fp32_from_fp32_groups[i];

            for (int j=0; j<model_group.size(); j++) {
                auto model_param = model_group[j];
                // model_param.grad = model_param.main_grad
                set_attr_from_this_to_that(model_param, "grad", model_param, "main_grad"); 
                
                if (!use_contiguous_buffers_in_local_ddp) {
                    // model_param.main_grad = NULL
                    set_attr_to_none(model_param, "main_grad");
                }
            }
        }
    }
}


void _multi_tensor_copy_this_to_that(std::vector<at::Tensor> these, std::vector<at::Tensor> those)
{
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(8)
    #endif
    for (int i=0; i<these.size(); i++) {
        those[i].copy_(these[i]);
    }
}


} // end of namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_copy_model_grads_to_main_grads", 
        &optimizer::_copy_model_grads_to_main_grads,
        "CPU version of copy_model_grads_to_main_grads.",
        py::call_guard<py::gil_scoped_release>());
        
    m.def("foreach_non_finite_check_and_unscale", 
        &optimizer::foreach_non_finite_check_and_unscale,
        "CPU version of at::_amp_foreach_non_finite_check_and_unscale_.",
        py::call_guard<py::gil_scoped_release>());

    m.def("_multi_tensor_copy_this_to_that", 
        &optimizer::_multi_tensor_copy_this_to_that,
        "CPU version of _multi_tensor_copy_this_to_that.",
        py::call_guard<py::gil_scoped_release>());
        
}