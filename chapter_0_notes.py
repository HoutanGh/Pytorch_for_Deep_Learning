import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.cuda.is_available())
# print(torch.__version__)

#tensors - torch.tensor(), torch.tensor(7).ndim, torch.tensor(7).item()

def my_tensor():
    tensor = torch.tensor([[1, 3], [1, 2], [1, 2]])   
    #tensor1 = tensor + 10
    tensor2 = torch.tensor([[[1, 3], [1, 2], [1, 2]]])
    
    print(tensor.ndim)
    print(tensor.shape)
    #print(tensor1)
    print(tensor2.ndim)

    return 0

# my_tensor()


def rand_tensor():
    random_tensor = torch.rand(size=(1, 5, 3, 4))

    print(random_tensor.shape)
    print(random_tensor.ndim)

    return 0

# rand_tensor()
# doesnt seem like you need to size=

def range_tensor():
    rangee = torch.arange(1, 10, 2)

    print(rangee)
    return 0

# range_tensor()

def minmax():
    x = torch.arange(0, 100, 10)
    print(x.argmin()) #index
    print(x.min())

    return 0

# minmax()

def reshapeviewstack():
    x = torch.arange(1, 100, 10)
    z = x.view(1, 10) # or an arbitrary amount of space of the original tensor that uses the same memory so changing z/x changhes x/z
    x_stacked = torch.stack([x, x, x, x], dim=1)

    print(x, z)
    print(x_stacked)

# reshapeviewstack()

def squeezepermut(): # squeeze removes a single dimension
    x = torch.arange(1, 8)
    x_reshaped = x.reshape(1, 7)
    x_squeeze = x_reshaped.squeeze()
    print(x)
    print(x_reshaped)
    print(x_squeeze)
    print(x_squeeze.unsqueeze(dim=0))

    x_permute = x.permute(1, 0)
    print(x_permute)

# squeezepermut()

def index():
    x = torch.arange(1, 10).reshape(1, 3, 3)
    print(x, x.device)

# index()

def device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor([1, 2, 3])
    tensor_gpu = tensor.to(device)
    print(tensor_gpu, tensor_gpu.device)
    tensor_cpu = tensor_gpu.cpu()
    print(tensor_cpu, tensor_cpu.device)

device()


    


