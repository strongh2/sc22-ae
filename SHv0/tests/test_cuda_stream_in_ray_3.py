import torch
import ray
import time

# ray.init(num_cpus=10, num_gpus=1)


@ray.remote
class Handler:
    def change_shared_tensor(self, num, obj_ref):
        print(f"---. {obj_ref}")
        obj_ref[0][0] = num
        print(f"---. {obj_ref}")


handler = Handler.options(num_gpus=1, num_cpus=10, max_concurrency=1).remote()

shared_tensor = torch.randn([2, 2])
print(shared_tensor)  # A

obj_ref = ray.put(shared_tensor)

futs = [handler.change_shared_tensor.remote(i, obj_ref) for i in range(10)]
ray.get(futs)

print(ray.get(obj_ref))  # A

"""
conclusion:

ray deepcopy param stensors between different actors and method invocations.


tensor([[ 1.2339, -0.9006],
        [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 0.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 2.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 3.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 4.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 5.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 6.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 7.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 8.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 1.2339, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])
[2m[36m(Handler pid=167)[0m ---. tensor([[ 9.0000, -0.9006],
[2m[36m(Handler pid=167)[0m         [-0.1162,  0.7115]])

"""
