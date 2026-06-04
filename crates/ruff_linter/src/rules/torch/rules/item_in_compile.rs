use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::in_compiled_function;

/// ## What it does
/// Checks for `.item()` calls inside a function decorated with
/// `@torch.compile`.
///
/// ## Why is this bad?
/// `.item()` materializes a tensor value as a Python scalar, which forces a
/// device-to-host synchronization and is opaque to TorchDynamo. The
/// compiler must break the graph and fall back to eager execution.
///
/// Try to keep the computation tensor-valued. When you need the Python
/// value, move the `.item()` call to the caller, after the compiled
/// function returns.
///
/// ## Example
/// ```python
/// import torch
///
///
/// @torch.compile
/// def f(x):
///     value = x.sum().item()  # Bad: graph break
///     return value + 1
///
///
/// @torch.compile
/// def f(x):
///     return x.sum() + 1  # Good: stays tensor-valued
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.compile` graph breaks](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ItemInCompile;

impl Violation for ItemInCompile {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`.item()` inside `@torch.compile` function causes a graph break".to_string()
    }
}

/// TORCH103
pub(crate) fn item_in_compile(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    let Expr::Attribute(ast::ExprAttribute { attr, .. }) = call.func.as_ref() else {
        return;
    };
    if attr.as_str() != "item" {
        return;
    }

    if !in_compiled_function(semantic) {
        return;
    }

    checker.report_diagnostic(ItemInCompile, call.range());
}
