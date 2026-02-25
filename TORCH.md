# TORCH - PyTorch Linter Rules for Ruff

## Motivation

Ruff does not currently support a plugin system for third-party lint rules
(see [astral-sh/ruff#283](https://github.com/astral-sh/ruff/issues/283)).
Before proposing to upstream dedicated PyTorch-specific rules, i want to propose
rules structure and potential, community agreed rules here.

The goal of the `TORCH` rule set is to catch common PyTorch anti-patterns that
lead to silent correctness bugs, performance regressions, or training failures.
These issues are notoriously hard to debug because they often produce no error,
just wrong results.

## Rule Numbering

| Range    | Category     | Description                                         |
| -------- | ------------ | --------------------------------------------------- |
| TORCH0XX | Tensors      | Tensor construction, access, and operations         |
| TORCH1XX | Compile      | `torch.compile` usage and graph-break hygiene       |
| TORCH2XX | nn.Module    | Model definition, initialization, and hooks         |
| TORCH3XX | DataLoading  | `DataLoader`, `Dataset`, and data pipeline mistakes |
| TORCH4XX | Distributed  | DDP, FSDP, and multi-process training               |
| TORCH5XX | Performance  | Avoidable performance pitfalls                      |
| TORCH6XX | Deprecations | Deprecated APIs and migration to modern equivalents |

## Implemented Rules

| Rule     | Name              | Description                                                          | Autofix |
| -------- | ----------------- | -------------------------------------------------------------------- | ------- |
| TORCH001 | TensorConstructor | `torch.Tensor()` creates uninitialized float32; use `torch.tensor()` | Yes     |
| TORCH002 | TensorDataAccess  | `tensor.data` bypasses autograd; prefer `tensor.detach()`            | No      |

## Proposed Rules

Rules below are widely acknowledged anti-patterns in the PyTorch community.
They are listed here as candidates for future implementation.

### Tensor Rules (TORCH0XX)

| Rule     | Description                                                          | Autofix |
| -------- | -------------------------------------------------------------------- | ------- |
| TORCH001 | `torch.Tensor()` creates uninitialized float32; use `torch.tensor()` | Yes     |
| TORCH002 | `tensor.data` bypasses autograd; prefer `tensor.detach()`            | No      |
| TORCH003 | Missing `force=True` when calling `.numpy()`                         | Yes     |
| TORCH004 | Missing `.detach()` before `.numpy()` / `.item()`                    | Yes     |
| TORCH005 | Missing `model.eval()` before inference                              | No      |
| TORCH006 | Use `torch.inference_mode` instead of `torch.no_grad`                | Yes     |
| TORCH007 | In-place op on a leaf tensor that requires grad                      | No      |
| TORCH008 | Explicit `dtype` missing in `torch.tensor()` call                    | No      |
| TORCH009 | Device mismatch: mixing CPU and CUDA tensors in an op                | No      |

### Compile Rules (TORCH1XX)

| Rule     | Short Description                                                   | Autofix |
| -------- | ------------------------------------------------------------------- | ------- |
| TORCH100 | `print()` inside a `torch.compile`-decorated function (graph break) | No      |
| TORCH101 | `try`/`except` inside a compiled function (graph break)             | No      |
| TORCH102 | Data-dependent `if` on a tensor value in compiled function          | No      |
| TORCH103 | Calling `.item()` inside a compiled function (graph break)          | No      |
| TORCH104 | Non-static `nn.Module` attribute access in compiled function        | No      |

### nn.Module Rules (TORCH2XX)

| Rule     | Short Description                                                                | Autofix |
| -------- | -------------------------------------------------------------------------------- | ------- |
| TORCH200 | `super().__init__()` missing in `nn.Module` subclass                             | No      |
| TORCH201 | Instance member assignment before `super().__init__()` in `nn.Module` subclass   | No      |
| TORCH202 | Layer assigned after `forward()` definition but not in `__init__`                | No      |
| TORCH203 | Using `self.modules()` when `self.children()` is intended                        | No      |
| TORCH204 | `nn.Module` stored in a plain `list`/`dict` instead of `ModuleList`/`ModuleDict` | No      |
| TORCH205 | Unused parameter: `nn.Parameter` defined but never used in `forward()`           | No      |

### DataLoading Rules (TORCH3XX)

| Rule     | Short Description                                                  | Autofix |
| -------- | ------------------------------------------------------------------ | ------- |
| TORCH300 | `drop_last=False` with DDP `DistributedSampler`                    | No      |
| TORCH301 | `__getitem__` opens file handle without closing it in `Dataset`    | No      |
| TORCH302 | Non-picklable object in `Dataset` breaks multi-worker `DataLoader` | No      |

### Distributed Rules (TORCH4XX)

| Rule     | Short Description                                                      | Autofix |
| -------- | ---------------------------------------------------------------------- | ------- |
| TORCH400 | Missing `dist.barrier()` before accessing shared file system resources | No      |
| TORCH401 | `model.to(device)` after `DistributedDataParallel` wrap                | No      |
| TORCH402 | `torch.compile(model)` before `DistributedDataParallel` wrap           | No      |
| TORCH403 | Unused parameters in DDP model (set `find_unused_parameters=True`)     | No      |
| TORCH404 | Accessing `.module` on a DDP Model.                                    | No      |

### Performance Rules (TORCH5XX)

| Rule     | Short Description                                                     | Autofix |
| -------- | --------------------------------------------------------------------- | ------- |
| TORCH500 | Gradient not zeroed before backward pass (`optimizer.zero_grad()`)    | No      |
| TORCH501 | Use `set_to_none=True` in `optimizer.zero_grad()`                     | Yes     |
| TORCH502 | Synchronous `.to(device)` in a hot loop; prefer pre-allocated buffers | No      |
| TORCH503 | Redundant `.contiguous()` call on already contiguous tensor           | No      |
| TORCH504 | `torch.cat` in a loop; accumulate in a list and cat once              | No      |
| TORCH505 | Using Python `for` loop over tensor dims; prefer vectorized ops       | No      |

### Deprecation Rules (TORCH6XX)

| Rule     | Short Description                                                      | Autofix |
| -------- | ---------------------------------------------------------------------- | ------- |
| TORCH600 | `torch.cuda.amp.autocast` replaced by `torch.amp.autocast("cuda")`     | Yes     |
| TORCH601 | `torch.cuda.amp.GradScaler` replaced by `torch.amp.GradScaler("cuda")` | Yes     |
| TORCH602 | `torch.nn.utils.clip_grad_value_` deprecated in 2.1                    | No      |
| TORCH603 | `torch.load` without `weights_only=True` (security / deprecation)      | Yes     |
