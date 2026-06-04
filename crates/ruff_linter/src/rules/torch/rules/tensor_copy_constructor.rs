use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::SemanticModel;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for `torch.tensor(t)` calls where `t` is itself produced by a
/// torch-returning call (e.g., another `torch.tensor(...)`, `torch.zeros`).
///
/// ## Why is this bad?
/// `torch.tensor()` always copies its argument, even when the argument is
/// already a tensor. The copy goes through a Python-level intermediate and is
/// significantly slower than `t.clone()`, which copies the data directly.
///
/// `t.clone()` preserves the source tensor's dtype and device, which
/// `torch.tensor(t)` does not, so the two are not strictly equivalent — but
/// when the source is already a tensor, `clone()` is almost always what was
/// intended.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.zeros(4)
///
/// # Bad: extra Python-side copy
/// y = torch.tensor(x)
///
/// # Good: direct clone
/// y = x.clone()
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html)
/// - [PyTorch documentation: `Tensor.clone`](https://pytorch.org/docs/stable/generated/torch.Tensor.clone.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TensorCopyConstructor;

impl Violation for TensorCopyConstructor {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.tensor(t)` copies an existing tensor; use `t.clone()` instead".to_string()
    }
}

/// TORCH010
pub(crate) fn tensor_copy_constructor(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.tensor") {
        return;
    }

    // Single positional argument, no keyword args we care about beyond it.
    let [arg] = call.arguments.args.as_ref() else {
        return;
    };

    let Expr::Name(name) = arg else {
        return;
    };

    if !name_binding_is_tensor(semantic, name) {
        return;
    }

    checker.report_diagnostic(TensorCopyConstructor, call.range());
}

/// Returns `true` if `name`'s binding was initialized from a call to a
/// `torch.*` function (a reasonable proxy for "this is a tensor").
fn name_binding_is_tensor(semantic: &SemanticModel, name: &ast::ExprName) -> bool {
    let Some(binding_id) = semantic.resolve_name(name) else {
        return false;
    };
    let binding = semantic.binding(binding_id);
    let Some(source) = binding.source else {
        return false;
    };

    let init = match semantic.statement(source) {
        Stmt::Assign(ast::StmtAssign { value, .. }) => value.as_ref(),
        Stmt::AnnAssign(ast::StmtAnnAssign {
            value: Some(value), ..
        }) => value.as_ref(),
        _ => return false,
    };

    let Expr::Call(init_call) = init else {
        return false;
    };

    semantic
        .resolve_qualified_name(&init_call.func)
        .is_some_and(|qname| qname.segments().first() == Some(&"torch"))
}
