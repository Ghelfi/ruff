"""Test cases for TORCH104: non-static nn.Module attribute access in compiled function."""

import torch


class M(torch.nn.Module):
    # Errors (should trigger TORCH104)

    @torch.compile
    def bad_assign(self, x):
        self.counter = 1
        return x

    @torch.compile
    def bad_aug_assign(self, x):
        self.counter += 1
        return x

    @torch.compile
    def bad_ann_assign(self, x):
        self.tag: int = 2
        return x

    @torch.compile(mode="default")
    def bad_factory(self, x):
        self.last_input = x
        return x

    @torch.compile
    def bad_multi_target(self, x):
        self.a = self.b = 1
        return x

    # should NOT trigger TORCH104

    def not_compiled(self, x):
        self.counter = 1
        return x

    @torch.compile
    def ok_no_self_mutation(self, x):
        y = x + 1
        return y

    @torch.compile
    def ok_local_attribute(self, x):
        other = SomethingElse()
        other.value = 1
        return x

    @torch.compile
    def ok_ann_no_value(self, x):
        # No value — pure annotation.
        self.tag: int
        return x
