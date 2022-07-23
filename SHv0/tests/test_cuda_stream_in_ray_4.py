import torch
import ray
import time

# ray.init(num_cpus=10, num_gpus=1)


@ray.remote
class Handler:
    def __init__(self, data):
        self.data = data

    def change_shared_tensor(self, num):
        print(f"{num} ---. {self.data}")
        self.data[0][0] = num
        print(f"{num} ---. {self.data}")


shared_tensor = torch.randn([2, 2])
print(shared_tensor)  # A

handler = Handler.options(num_gpus=1, num_cpus=10, max_concurrency=1).remote(
    shared_tensor
)

futs = [handler.change_shared_tensor.remote(i) for i in range(10)]
ray.get(futs)

print(shared_tensor)  # A

"""
conclusion:

ray deepcopy param stensors between different actors and method invocations.

ray actor can use same one tensor while invoking its own methods.

tensor([[-0.2613, -0.8304],
        [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 0 ---. tensor([[-0.2613, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 0 ---. tensor([[ 0.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 1 ---. tensor([[ 0.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 1 ---. tensor([[ 1.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 2 ---. tensor([[ 1.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 2 ---. tensor([[ 2.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 3 ---. tensor([[ 2.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 3 ---. tensor([[ 3.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 4 ---. tensor([[ 3.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 4 ---. tensor([[ 4.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 5 ---. tensor([[ 4.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 5 ---. tensor([[ 5.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 6 ---. tensor([[ 5.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 6 ---. tensor([[ 6.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 7 ---. tensor([[ 6.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 7 ---. tensor([[ 7.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 8 ---. tensor([[ 7.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 8 ---. tensor([[ 8.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 9 ---. tensor([[ 8.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
[2m[36m(Handler pid=257)[0m 9 ---. tensor([[ 9.0000, -0.8304],
[2m[36m(Handler pid=257)[0m         [-0.8810,  0.2488]])
tensor([[-0.2613, -0.8304],
        [-0.8810,  0.2488]])


"""
