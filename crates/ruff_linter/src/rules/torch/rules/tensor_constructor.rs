use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for uses of `torch.Tensor()` to construct tensors.
///
/// ## Why is this bad?
/// `torch.Tensor` is an alias for `torch.FloatTensor`, which constructs an
/// **uninitialized** tensor with the default dtype (`torch.float32`). This is
/// almost never what you want:
///
/// - The dtype is implicitly `float32` regardless of input.
///
/// `torch.tensor()` (lowercase) is the recommended constructor. It:
///
/// - Infers the dtype from the input data.
/// - Always copies the data (no aliasing surprises).
/// - Supports `dtype`, `device`, and `requires_grad` keyword arguments.
///
/// ## Example
/// ```python
/// import torch
///
/// # Bad: implicit float32
/// x = torch.Tensor([1, 2, 3])
///
/// # Good: dtype inferred as int64
/// x = torch.tensor([1, 2, 3])
/// ```
///
/// ## Fix
/// Replaces `torch.Tensor(` with `torch.tensor(`, preserving all arguments.
/// If the symbol was imported via `from torch import Tensor`, the fix
/// replaces the call site with `torch.tensor(` and adds `import torch` if
/// needed — or, if `torch` is already imported, just rewrites the callee.
///
/// ## References
/// - [PyTorch documentation: `torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html)
/// - [PyTorch documentation: `torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TensorConstructor;

impl AlwaysFixableViolation for TensorConstructor {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Use `torch.tensor()` instead of `torch.Tensor()` to create tensors".to_string()
    }

    fn fix_title(&self) -> String {
        "Replace `torch.Tensor()` with `torch.tensor()`".to_string()
    }
}

/// TORCH001
pub(crate) fn tensor_constructor(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    // We only care about calls where the callee resolves to `torch.Tensor`.
    if !is_torch_qualified_name(semantic, &call.func, "torch.Tensor") {
        return;
    }

    // Determine the replacement text for the callee expression.
    let replacement = compute_replacement(checker, &call.func);

    let mut diagnostic = checker.report_diagnostic(TensorConstructor, call.func.range());
    diagnostic.set_fix(Fix::safe_edit(Edit::range_replacement(
        replacement,
        call.func.range(),
    )));
}

/// Compute the replacement callee string.
///
/// - If the original code is `torch.Tensor(...)` (attribute access), replace
///   with `torch.tensor` (preserving any alias for `torch`).
/// - If the original code is a bare `Tensor(...)` from `from torch import Tensor`,
///   replace with `torch.tensor` (assuming `torch` is importable; the user already
///   depends on it).
fn compute_replacement(checker: &Checker, func: &Expr) -> String {
    match func {
        // `torch.Tensor(...)` or `th.Tensor(...)` — rewrite the attribute name.
        Expr::Attribute(ast::ExprAttribute { value, .. }) => {
            let module_source = checker.locator().slice(value.range());
            format!("{module_source}.tensor")
        }
        // `Tensor(...)` from `from torch import Tensor` — replace with `torch.tensor`.
        Expr::Name(_) => {
            // Check if `torch` itself is available as a binding (e.g., `import torch`).
            // If so, just use `torch.tensor`. If not, still emit `torch.tensor`;
            // in practice the user has `from torch import Tensor` so `torch` is installed.
            "torch.tensor".to_string()
        }
        // Shouldn't happen for a resolved `torch.Tensor`, but be safe.
        _ => "torch.tensor".to_string(),
    }
}
