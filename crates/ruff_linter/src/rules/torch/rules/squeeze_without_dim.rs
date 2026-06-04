use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for `.squeeze()` calls that do not specify a `dim` argument.
///
/// ## Why is this bad?
/// `.squeeze()` without a `dim` argument removes *all* dimensions of size 1.
/// Whenever the input tensor's shape changes at runtime (e.g., a batch
/// dimension is 1 by coincidence), this silently drops extra dimensions and
/// the rest of the pipeline starts seeing a tensor of unexpected rank.
///
/// Always pass `dim=` so the squeeze is restricted to the dimension you
/// actually mean to remove.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.zeros(1, 3, 1)
///
/// # Bad: drops both the leading and trailing 1 dimensions
/// y = x.squeeze()
///
/// # Good: only drops the trailing dimension
/// y = x.squeeze(dim=-1)
/// ```
///
/// ## References
/// - [PyTorch documentation: `Tensor.squeeze`](https://pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct SqueezeWithoutDim;

impl Violation for SqueezeWithoutDim {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`.squeeze()` without `dim` may silently drop unexpected dimensions".to_string()
    }
}

/// TORCH013
pub(crate) fn squeeze_without_dim(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(ast::ExprAttribute { attr, .. }) = call.func.as_ref() else {
        return;
    };

    if attr.as_str() != "squeeze" {
        return;
    }

    if !call.arguments.args.is_empty() || call.arguments.find_keyword("dim").is_some() {
        return;
    }

    checker.report_diagnostic(SqueezeWithoutDim, call.range());
}
