"""Test cases for TORCH005: Missing model.eval() before inference."""

import torch


# Errors (should trigger TORCH005)


def predict_bad(model, x):
    # No .eval() anywhere in the enclosing function.
    with torch.no_grad():
        return model(x)


def predict_bad_inference(model, x):
    with torch.inference_mode():
        return model(x)


def predict_bad_multi_item(model, x):
    with torch.no_grad(), open("/tmp/log", "w") as f:
        return model(x)


def predict_bad_nested_eval(model, x):
    # .eval() inside a nested function does not protect the outer scope.
    def helper():
        model.eval()

    with torch.no_grad():
        return model(x)


# should NOT trigger TORCH005


def predict_ok(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def predict_ok_after(model, x):
    # .eval() appearing anywhere in the function is treated as sufficient.
    with torch.no_grad():
        out = model(x)
    model.eval()
    return out


def not_no_grad():
    # Different context manager.
    with open("/tmp/foo") as f:
        return f.read()


# Module-level inference blocks are not checked.
with torch.no_grad():
    pass
