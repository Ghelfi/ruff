use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for `.clone()` calls on a tensor whose binding was created with
/// `requires_grad=True`, without a `.detach()` in the chain.
///
/// ## Why is this bad?
/// `tensor.clone()` is a differentiable operation: gradients flow back through
/// the clone into the original tensor. When the cloned tensor is being copied
/// for inspection, snapshotting, or to pass into an external library, this
/// extra graph connection is usually unintentional and keeps the original
/// graph alive longer than necessary (a memory leak).
///
/// Use `.detach().clone()` (or `.clone().detach()`) when the clone should be
/// a standalone tensor outside autograd.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0], requires_grad=True)
///
/// # Bad: clone still participates in autograd
/// snapshot = x.clone()
///
/// # Good: detached clone
/// snapshot = x.detach().clone()
/// ```
///
/// ## References
/// - [PyTorch documentation: `Tensor.clone`](https://pytorch.org/docs/stable/generated/torch.Tensor.clone.html)
/// - [PyTorch documentation: `Tensor.detach`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct CloneWithoutDetach;

impl Violation for CloneWithoutDetach {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`.clone()` on a grad-tracked tensor without `.detach()`".to_string()
    }
}

/// TORCH011
pub(crate) fn clone_without_detach(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(ast::ExprAttribute { attr, value, .. }) = call.func.as_ref() else {
        return;
    };

    if attr.as_str() != "clone" {
        return;
    }

    let Expr::Name(name) = value.as_ref() else {
        return;
    };

    if !binding_requires_grad(semantic, name) {
        return;
    }

    checker.report_diagnostic(CloneWithoutDetach, call.range());
}

/// Returns `true` if `name`'s binding initializer is a call with
/// `requires_grad=True`.
fn binding_requires_grad(semantic: &SemanticModel, name: &ast::ExprName) -> bool {
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

    let Expr::Call(call) = init else {
        return false;
    };

    let Some(keyword) = call.arguments.find_keyword("requires_grad") else {
        return false;
    };

    matches!(&keyword.value, Expr::BooleanLiteral(b) if b.value)
}
