
import torch
import ray

ray.init(num_cpus=2, num_gpus=1)

@ray.remote
def func(obj_ref):
    # tensor = ray.get([obj_ref])
    print(obj_ref)
    obj_ref[0][0] = -1
    return obj_ref

tensor_cpu = torch.randn([100, 100])
obj_ref = ray.put(tensor_cpu)

tensor_cpu[0][0] = 2

tensor_cpu_ = ray.get([func.remote(obj_ref)])

print(tensor_cpu)
# print(tensor_cpu_)


# tensor_cuda = torch.randn([100, 100]).cuda()


ray.shutdown()