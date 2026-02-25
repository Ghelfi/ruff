use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for accesses to `.data` on `PyTorch` tensors.
///
/// ## Why is this bad?
/// `tensor.data` returns a new tensor that shares storage with the original
/// but is **not tracked by autograd**. Modifications through `.data` silently
/// break gradient computation, leading to incorrect training results that are
/// difficult to debug.
///
/// Use `tensor.detach()` instead, which also returns a tensor that shares
/// storage but makes the detachment from the computation graph explicit and
/// safer to reason about.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0], requires_grad=True)
///
/// # Bad: bypasses autograd
/// y = x.data
///
/// # Good: explicit detach
/// y = x.detach()
/// ```
///
/// ## References
/// - [PyTorch documentation: `Tensor.data`](https://pytorch.org/docs/stable/tensors.html)
/// - [PyTorch documentation: `Tensor.detach`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TensorDataAccess;

impl Violation for TensorDataAccess {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Avoid accessing `tensor.data`; use `tensor.detach()` instead".to_string()
    }
}

/// TORCH002
pub(crate) fn tensor_data_access(checker: &Checker, attribute: &ast::ExprAttribute) {
    let semantic = checker.semantic();

    // Fast path: skip if torch was never imported.
    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    // Only flag load context (not `x.data = ...`).
    if !attribute.ctx.is_load() {
        return;
    }

    // Only interested in `.data`.
    if attribute.attr.as_str() != "data" {
        return;
    }

    // Avoid flagging on function calls (e.g., `x.data()`).
    if semantic
        .current_expression_parent()
        .is_some_and(Expr::is_call_expr)
    {
        return;
    }

    // Exclude module-level attribute access like `torch.utils.data`.
    if semantic
        .resolve_qualified_name(&Expr::Attribute(attribute.clone()))
        .is_some()
    {
        return;
    }

    // Exclude `self.data` and `cls.data` — common in non-tensor contexts.
    if let Expr::Name(name) = attribute.value.as_ref() {
        if matches!(name.id.as_str(), "self" | "cls") {
            return;
        }
    }

    // Exclude literals and other irrelevant expressions.
    if matches!(
        attribute.value.as_ref(),
        Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::NumberLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::NoneLiteral(_)
            | Expr::EllipsisLiteral(_)
            | Expr::Tuple(_)
            | Expr::List(_)
            | Expr::Set(_)
            | Expr::Dict(_)
    ) {
        return;
    }

    checker.report_diagnostic(TensorDataAccess, attribute.range());
}
