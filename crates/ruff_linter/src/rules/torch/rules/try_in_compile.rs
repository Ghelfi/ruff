use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::in_compiled_function;

/// ## What it does
/// Checks for `try`/`except` blocks inside a function decorated with
/// `@torch.compile`.
///
/// ## Why is this bad?
/// `torch.compile` (TorchDynamo) does not trace through Python exception
/// handling. Encountering a `try`/`except` block triggers a graph break,
/// splitting the compiled region and falling back to eager execution for
/// the handler.
///
/// Either move the exception-handling logic outside of the compiled
/// function or rewrite the code to avoid the exception path on the hot
/// path.
///
/// ## Example
/// ```python
/// import torch
///
/// @torch.compile
/// def f(x):
///     try:  # Bad: graph break
///         return 1.0 / x
///     except ZeroDivisionError:
///         return torch.zeros_like(x)
///
/// def f(x):
///     return torch.where(x == 0, torch.zeros_like(x), 1.0 / x)  # Good
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.compile` graph breaks](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TryInCompile;

impl Violation for TryInCompile {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`try`/`except` inside `@torch.compile` function causes a graph break".to_string()
    }
}

/// TORCH101
pub(crate) fn try_in_compile(checker: &Checker, try_stmt: &ast::StmtTry) {
    let semantic = checker.semantic();

    if !in_compiled_function(semantic) {
        return;
    }

    checker.report_diagnostic(TryInCompile, try_stmt.range());
}
