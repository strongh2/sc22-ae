import torch
import ray
import time

# ray.init(num_cpus=10, num_gpus=1)


@ray.remote
class Handler:
    def print_cuda_stream(self):
        print(f"start print_cuda_stream: {torch.cuda.current_stream()}", flush=True)
        for i in range(500000):
            pass
        print(f"end print_cuda_stream")

    def set_cuda_stream(self, num):
        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            for i in range(100000):
                pass
            print(
                f"{num} - set_cuda_stream: {torch.cuda.current_stream()}",
                flush=True,
            )


handler = Handler.options(num_gpus=1, num_cpus=10, max_concurrency=10).remote()

futs = [handler.set_cuda_stream.remote(i) for i in range(10)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(15, 20)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(25, 30)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(35, 40)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(45, 50)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(55, 60)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(65, 70)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
futs += [handler.set_cuda_stream.remote(i) for i in range(75, 80)]
futs += [handler.print_cuda_stream.remote() for i in range(10)]
ray.get(futs)

"""
conclusion:

torch.cuda.current_stream is thread-local variable.

"""
