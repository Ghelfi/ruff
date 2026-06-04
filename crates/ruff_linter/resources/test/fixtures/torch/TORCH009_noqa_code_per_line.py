import torch

bad1 = a.cuda() + b.cpu()  # noqa: TORCH009
bad2 = a.cuda() + b.cpu()
