use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::in_compiled_function;

/// ## What it does
/// Checks for `print()` calls inside a function decorated with
/// `@torch.compile`.
///
/// ## Why is this bad?
/// `torch.compile` traces the function body into a TorchDynamo graph. Side
/// effects like `print` cannot be captured in the graph, so the tracer is
/// forced to break the graph at the print call, falling back to eager
/// execution for that part. Graph breaks reduce the performance benefit of
/// compilation.
///
/// Either remove the `print()` call or move it outside the compiled
/// function. For debug output during tracing, prefer `torch._dynamo.config`
/// debug flags.
///
/// ## Example
/// ```python
/// import torch
///
/// @torch.compile
/// def f(x):
///     print(x)  # Bad: graph break
///     return x + 1
///
/// @torch.compile
/// def f(x):
///     return x + 1  # Good: no graph break
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.compile` graph breaks](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct PrintInCompile;

impl Violation for PrintInCompile {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`print()` inside `@torch.compile` function causes a graph break".to_string()
    }
}

/// TORCH100
pub(crate) fn print_in_compile(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.match_builtin_expr(&call.func, "print") {
        return;
    }

    if !in_compiled_function(semantic) {
        return;
    }

    checker.report_diagnostic(PrintInCompile, call.range());
}
