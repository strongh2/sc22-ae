import torch
import ray
import time


@ray.remote
class Handler:
    def set_cuda_stream(self, num):
        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            for i in range(10000000):
                pass


handler = Handler.options(num_gpus=1, max_concurrency=10).remote()

start_time = time.time()

futs = [handler.set_cuda_stream.remote(i) for i in range(10)]
ray.get(futs)

end_time = time.time()

print(end_time - start_time)


start_time = time.time()

futs = [handler.set_cuda_stream.remote(i) for i in range(10)]
ray.get(futs)

end_time = time.time()

print(end_time - start_time)

start_time = time.time()

[ray.get(handler.set_cuda_stream.remote(i)) for i in range(10)]

end_time = time.time()

print(end_time - start_time)


"""
conclusion:

if only python code, no parallelism in threaded actor.
The similar time overhead in max_concurrency=1 and max_concurrency=10
"""
