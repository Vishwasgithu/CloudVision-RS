import torch

print("PyTorch:", torch.__version__)

x = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(x)
print(x.shape)
print(type(x))
print(x.device)