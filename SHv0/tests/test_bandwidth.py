import time
import torch

#should be 40 GB
#torch.rand([1024, 1024, 1024, 10])

# g means g GB
mock_data = [
    lambda g: torch.rand([1024, 1024, 256, g]),
    lambda g: torch.rand([1024, 512, 512, g]),
    lambda g: torch.rand([512, 512, 512, 2, g]),
    lambda g: torch.rand([2048, 512, 256, g]),
    lambda g: torch.rand([4096, 256, 256, g]),
    lambda g: torch.rand([128, 8, 128, 8, 128, 2, g]),
]

cuda_device = torch.device('cuda:0')
cpu_device = torch.device('cpu') 


def cpu_to_gpu(tensor, cuda_tensor):
    assert tensor.device == cpu_device
    assert cuda_tensor.device == cuda_device

    start_time = time.time()
    
    cuda_tensor.copy_(tensor)
    #cuda_tensor = tensor.cuda()
    
    end_time = time.time()
    
    duration = end_time - start_time
    size = 4 / 1024 / 1024 / 1024 
    for dim in tensor.size():
        size *= dim
    speed = size / duration

    print('speed c2g (GB/s): ', speed, '; size (GB): ', size, '; duration (s): ', duration, '; shape: ', tensor.size())


def gpu_to_cpu(tensor, cpu_tensor):
    assert tensor.device == cuda_device
    assert cpu_tensor.device == cpu_device

    start_time = time.time()

    cpu_tensor.copy_(tensor)
    #cpu_tensor = tensor.cpu()
    
    end_time = time.time()
    
    duration = end_time - start_time

    size = 4 / 1024 / 1024 / 1024 
    for dim in tensor.size():
        size *= dim
    speed = size / duration

    print('speed g2c (GB/s): ', speed, '; size (GB): ', size, '; duration (s): ', duration, '; shape: ', tensor.size())


for i in range(1, 20):
    for j in range(6):
        for k in range(3):
            gpu_to_cpu(mock_data[j](i).cuda(), mock_data[j](i).pin_memory())

for i in range(1, 20):
    for j in range(6):
        for k in range(3):
            cpu_to_gpu(mock_data[j](i).pin_memory(), mock_data[j](i).cuda())


#for i in range(1, 20):
#    for j in range(6):
#        for k in range(3):
#            cpu_to_gpu(mock_data[j](i))

