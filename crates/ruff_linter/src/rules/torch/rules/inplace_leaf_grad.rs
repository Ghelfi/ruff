use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for in-place tensor operations (methods ending in `_`) applied to
/// a name whose binding was created with `requires_grad=True`.
///
/// ## Why is this bad?
/// PyTorch does not allow in-place operations on a leaf tensor that requires
/// grad — it raises a `RuntimeError` at runtime. Even when wrapped in a
/// `with torch.no_grad():` block (which silences the error), modifying a
/// leaf parameter in place can desynchronize the version counter and produce
/// silently incorrect gradients on the next backward pass.
///
/// Use the out-of-place variant (e.g., `x = x + y` instead of `x.add_(y)`),
/// wrap the assignment in `torch.no_grad()`, or call `.detach()` first if the
/// in-place mutation is genuinely intended.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0], requires_grad=True)
///
/// # Bad: in-place op on a leaf tensor that requires grad
/// x.add_(1.0)
///
/// # Good: use the out-of-place op
/// x = x + 1.0
///
/// # Or: explicit detach if the mutation is intentional
/// x.detach().add_(1.0)
/// ```
///
/// ## References
/// - [PyTorch documentation: leaf tensors](https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct InplaceLeafGrad {
    method: String,
}

impl Violation for InplaceLeafGrad {
    #[derive_message_formats]
    fn message(&self) -> String {
        let InplaceLeafGrad { method } = self;
        format!("In-place op `.{method}()` on a leaf tensor that requires grad")
    }
}

/// TORCH007
pub(crate) fn inplace_leaf_grad(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(ast::ExprAttribute { attr, value, .. }) = call.func.as_ref() else {
        return;
    };

    // PyTorch convention: in-place tensor ops end in a single underscore.
    // Skip dunder methods (`__init__`, etc.) and methods starting with `_`.
    let method = attr.as_str();
    if !is_inplace_method(method) {
        return;
    }

    let Expr::Name(name) = value.as_ref() else {
        return;
    };

    if !binding_requires_grad(semantic, name) {
        return;
    }

    checker.report_diagnostic(
        InplaceLeafGrad {
            method: method.to_string(),
        },
        call.range(),
    );
}

fn is_inplace_method(name: &str) -> bool {
    name.ends_with('_') && !name.starts_with('_') && name.len() > 1
}

/// Returns `true` if `name` resolves to a binding whose initializer is a call
/// containing `requires_grad=True` (e.g., `torch.tensor(..., requires_grad=True)`).
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
