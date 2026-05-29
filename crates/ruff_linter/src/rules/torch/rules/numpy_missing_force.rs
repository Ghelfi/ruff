use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::fix::edits::add_argument;
use crate::{AlwaysFixableViolation, Applicability, Fix};

/// ## What it does
/// Checks for calls to `.numpy()` on PyTorch tensors that are missing
/// `force=True`.
///
/// ## Why is this bad?
/// Calling `.numpy()` without `force=True` will raise a `RuntimeError` if
/// the tensor requires grad, is on a non-CPU device, or is a conjugate or
/// negative-bit view. Adding `force=True` (introduced in PyTorch 2.1)
/// automatically detaches and moves the tensor to CPU before converting,
/// making the call robust regardless of tensor state.
///
/// Without `force=True`, callers must manually chain `.detach().cpu()`
/// before `.numpy()`, which is verbose and easy to forget.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0], requires_grad=True)
///
/// # Bad: will raise RuntimeError
/// arr = x.numpy()
///
/// # Good: handles grad / device automatically
/// arr = x.numpy(force=True)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because it changes runtime behavior: calls that
/// would have raised a `RuntimeError` (grad / CUDA / conjugate views) will
/// instead silently succeed by detaching and copying to CPU.
///
/// ## References
/// - [PyTorch documentation: `Tensor.numpy`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct NumpyMissingForce;

impl AlwaysFixableViolation for NumpyMissingForce {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Missing `force=True` when calling `.numpy()`".to_string()
    }

    fn fix_title(&self) -> String {
        "Add `force=True` argument".to_string()
    }
}

/// TORCH003
pub(crate) fn numpy_missing_force(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    // Fast path: skip if torch was never imported.
    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    // We only care about method calls like `expr.numpy(...)`.
    let Expr::Attribute(ast::ExprAttribute { attr, .. }) = call.func.as_ref() else {
        return;
    };

    if attr.as_str() != "numpy" {
        return;
    }

    // Skip if a `force=` argument is already present in any form.
    // An explicit `force=False` (or `force=some_expr`) is a deliberate user
    // choice; do not lint against it, and never duplicate the keyword.
    if call.arguments.find_keyword("force").is_some() {
        return;
    }

    let mut diagnostic = checker.report_diagnostic(NumpyMissingForce, call.range());
    diagnostic.set_fix(Fix::applicable_edit(
        add_argument("force=True", &call.arguments, checker.tokens()),
        Applicability::Unsafe,
    ));
}
