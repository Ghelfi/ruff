use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::{AlwaysFixableViolation, Applicability, Edit, Fix};

/// ## What it does
/// Checks for calls to `.numpy()` or `.item()` on `PyTorch` tensors that are
/// not preceded by a `.detach()` in the same expression chain.
///
/// ## Why is this bad?
/// Calling `.numpy()` on a tensor that requires grad raises a `RuntimeError`,
/// and `.item()` / `.tolist()` silently extract values that are no longer
/// tracked by autograd. Detaching first makes the intent explicit and avoids
/// surprising runtime errors when the upstream tensor happens to require
/// grad.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0], requires_grad=True)
///
/// # Bad: may raise RuntimeError, or silently break grad tracking
/// arr = x.numpy()
/// value = x.item()
///
/// # Good: explicit detach before extracting values
/// arr = x.detach().numpy()
/// value = x.detach().item()
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because it changes runtime behavior: calls that
/// would have raised a `RuntimeError` will instead silently succeed after
/// detaching from the autograd graph.
///
/// ## References
/// - [PyTorch documentation: `Tensor.detach`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
/// - [PyTorch documentation: `Tensor.numpy`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)
/// - [PyTorch documentation: `Tensor.item`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct MissingDetach {
    method: String,
}

impl AlwaysFixableViolation for MissingDetach {
    #[derive_message_formats]
    fn message(&self) -> String {
        let MissingDetach { method } = self;
        format!("Missing `.detach()` before `.{method}()`")
    }

    fn fix_title(&self) -> String {
        "Insert `.detach()` before the call".to_string()
    }
}

/// TORCH004
pub(crate) fn missing_detach(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(ast::ExprAttribute { attr, value, .. }) = call.func.as_ref() else {
        return;
    };

    let method = attr.as_str();
    if !matches!(method, "numpy" | "item" | "tolist") {
        return;
    }

    // Walk the receiver chain. If `.detach()` already appears, skip.
    if receiver_chain_contains_detach(value) {
        return;
    }

    let mut diagnostic = checker.report_diagnostic(
        MissingDetach {
            method: method.to_string(),
        },
        call.range(),
    );
    // Insert `.detach()` immediately after the receiver expression.
    diagnostic.set_fix(Fix::applicable_edit(
        Edit::insertion(".detach()".to_string(), value.end()),
        Applicability::Unsafe,
    ));
}

/// Returns `true` if any attribute or call in the receiver chain rooted at
/// `expr` is `.detach(...)`.
fn receiver_chain_contains_detach(expr: &Expr) -> bool {
    let mut current = expr;
    loop {
        match current {
            Expr::Call(call) => {
                if let Expr::Attribute(ast::ExprAttribute { attr, value, .. }) = call.func.as_ref()
                {
                    if attr.as_str() == "detach" {
                        return true;
                    }
                    current = value;
                } else {
                    return false;
                }
            }
            Expr::Attribute(ast::ExprAttribute { value, .. }) => {
                current = value;
            }
            _ => return false,
        }
    }
}
