"""Test cases for TORCH605: `torch.hub.load` without `trust_repo`."""

import torch

# Errors (should trigger TORCH605)
torch.hub.load("pytorch/vision", "resnet50")
torch.hub.load("pytorch/vision", "resnet50", pretrained=True)

# Should NOT trigger TORCH605
torch.hub.load("pytorch/vision", "resnet50", trust_repo=True)
torch.hub.load("pytorch/vision", "resnet50", trust_repo=False)
torch.hub.load("pytorch/vision", "resnet50", trust_repo="check")
