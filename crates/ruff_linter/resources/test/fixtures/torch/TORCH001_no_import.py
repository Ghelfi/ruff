"""Test case: no torch import at all. Nothing should trigger."""

# No import of torch — these should NOT trigger.
# (If someone has a local `Tensor` or `torch` that isn't the real torch module.)


class torch:
    @staticmethod
    def Tensor():
        pass


# This is a user-defined `torch`, not the real module.
x = torch.Tensor()  # OK — not the real torch


class Tensor:
    pass


my_tensor = Tensor()  # OK — not torch.Tensor