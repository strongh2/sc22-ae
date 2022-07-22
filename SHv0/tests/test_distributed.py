
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp as ReduceOp
import os

os.environ['MASTER_PORT']= str(63451)
os.environ['MASTER_ADDR']='127.0.0.1'

dist.init_process_group(backend='nccl', rank=0, world_size=1)

pg = dist.distributed_c10d._get_default_group()

print(dir(pg))


tensor = torch.arange(2, dtype=torch.int64) + 1
print(tensor.device, tensor)
dist.all_reduce(tensor, op=ReduceOp.SUM)

tensor = tensor.cuda()
print(tensor.device, tensor)
dist.all_reduce(tensor, op=ReduceOp.SUM)
