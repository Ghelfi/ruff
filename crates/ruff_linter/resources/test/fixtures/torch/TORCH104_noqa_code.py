# ruff: noqa: TORCH104

import torch


class M(torch.nn.Module):
    @torch.compile
    def forward(self, x):
        self.counter += 1
        return x
