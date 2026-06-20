"""Test cases for TORCH200: missing `super().__init__()` in `nn.Module` subclass."""

import torch
import torch.nn as nn


# Errors (should trigger TORCH200)


class BadNoSuper(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 10)


class BadFullyQualified(torch.nn.Module):
    def __init__(self, n):
        self.n = n


class Base(nn.Module):
    def __init__(self):
        super().__init__()


class BadInherited(Base):
    # Subclass of an `nn.Module` subclass that overrides `__init__`.
    def __init__(self):
        self.x = 1


# should NOT trigger TORCH200


class OkWithSuper(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)


class OkSuperWithArgs(nn.Module):
    def __init__(self):
        super(OkSuperWithArgs, self).__init__()
        self.fc = nn.Linear(10, 10)


class OkNoInit(nn.Module):
    # Inherits `nn.Module.__init__`, nothing to flag.
    def forward(self, x):
        return x


class NotAModule:
    def __init__(self):
        self.x = 1
