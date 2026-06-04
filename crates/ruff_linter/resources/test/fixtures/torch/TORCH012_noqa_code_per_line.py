import torch

x = torch.tensor([1.0])
y = x.cuda()  # noqa: TORCH012
z = x.cuda()
