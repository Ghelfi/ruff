# ruff: noqa: TORCH009

import torch

bad = a.cuda() + b.cpu()
