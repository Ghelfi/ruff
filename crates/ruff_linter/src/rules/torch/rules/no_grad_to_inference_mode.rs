use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for uses of `torch.no_grad()` and suggests `torch.inference_mode()`
/// instead.
///
/// ## Why is this bad?
/// `torch.inference_mode()` was introduced in PyTorch 1.9 as a strictly more
/// performant alternative to `torch.no_grad()` for inference-only code. It
/// disables view tracking and version counter bumps in addition to autograd,
/// which removes overhead that `torch.no_grad()` still pays.
///
/// `inference_mode()` is appropriate whenever the resulting tensors will not
/// participate in autograd at all (the most common case for inference).
///
/// ## Example
/// ```python
/// import torch
///
/// # Bad
/// with torch.no_grad():
///     out = model(x)
///
/// # Good
/// with torch.inference_mode():
///     out = model(x)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because `inference_mode()` is stricter than
/// `no_grad()`: tensors produced inside it cannot later be used in autograd
/// computations (e.g., as part of a loss or backward call). Code that relies
/// on this will raise a `RuntimeError` after the fix.
///
/// ## References
/// - [PyTorch documentation: `torch.inference_mode`](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)
/// - [PyTorch documentation: `torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct NoGradToInferenceMode;

impl AlwaysFixableViolation for NoGradToInferenceMode {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Use `torch.inference_mode()` instead of `torch.no_grad()`".to_string()
    }

    fn fix_title(&self) -> String {
        "Replace `torch.no_grad` with `torch.inference_mode`".to_string()
    }
}

/// TORCH006
pub(crate) fn no_grad_to_inference_mode(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.no_grad") {
        return;
    }

    let replacement = match call.func.as_ref() {
        // `torch.no_grad(...)` or `th.no_grad(...)` — keep the module alias.
        Expr::Attribute(ast::ExprAttribute { value, .. }) => {
            let module_source = checker.locator().slice(value.range());
            format!("{module_source}.inference_mode")
        }
        // `from torch import no_grad; no_grad(...)` — use qualified form.
        Expr::Name(_) => "torch.inference_mode".to_string(),
        _ => return,
    };

    let mut diagnostic = checker.report_diagnostic(NoGradToInferenceMode, call.func.range());
    diagnostic.set_fix(Fix::unsafe_edit(Edit::range_replacement(
        replacement,
        call.func.range(),
    )));
}
