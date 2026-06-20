"""Test cases for TORCH201: instance member assignment before `super().__init__()`."""

import torch.nn as nn


# Errors (should trigger TORCH201)


class BadAssign(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 10)
        super().__init__()


class BadMultiple(nn.Module):
    def __init__(self):
        self.a = 1
        self.b: int = 2
        self.c = 3
        super().__init__()


class BadAugAssign(nn.Module):
    def __init__(self):
        self.counter = 0
        self.counter += 1
        super().__init__()


# should NOT trigger TORCH201


class OkOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)


class OkLocalBeforeSuper(nn.Module):
    def __init__(self):
        x = 1  # Local variable, not a `self` attribute.
        super().__init__()
        self.x = x


class OkNoSuper(nn.Module):
    # Missing `super().__init__()` entirely is TORCH200's concern.
    def __init__(self):
        self.fc = nn.Linear(10, 10)


class NotAModule:
    def __init__(self):
        self.x = 1
        super().__init__()
