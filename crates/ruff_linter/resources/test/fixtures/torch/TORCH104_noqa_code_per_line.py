import torch


class M(torch.nn.Module):
    @torch.compile
    def forward(self, x):
        self.counter = 1  # noqa: TORCH104
        self.other = 2
        return x
