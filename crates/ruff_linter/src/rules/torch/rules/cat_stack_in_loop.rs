use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Stmt};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.cat` or `torch.stack` inside a `for` or
/// `while` loop body.
///
/// ## Why is this bad?
/// `torch.cat` and `torch.stack` allocate a new tensor large enough to hold
/// every input, then copy the data in. Calling them in a loop yields a
/// quadratic-time accumulation pattern: each iteration re-allocates and
/// re-copies all previously accumulated elements.
///
/// A common idiom is to append the per-iteration tensors to a Python list
/// and call `torch.cat` / `torch.stack` once after the loop.
///
/// ## Example
/// ```python
/// import torch
///
/// result = torch.empty(0)
/// for x in xs:
///     result = torch.cat([result, x])
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// parts = []
/// for x in xs:
///     parts.append(x)
/// result = torch.cat(parts)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.cat`](https://pytorch.org/docs/stable/generated/torch.cat.html)
/// - [PyTorch documentation: `torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct CatStackInLoop {
    name: &'static str,
}

impl Violation for CatStackInLoop {
    #[derive_message_formats]
    fn message(&self) -> String {
        let CatStackInLoop { name } = self;
        format!(
            "`torch.{name}` in a loop reallocates each iteration; accumulate in a list and call `torch.{name}` once"
        )
    }
}

/// TORCH504
pub(crate) fn cat_stack_in_loop(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    let name = if is_torch_qualified_name(semantic, &call.func, "torch.cat") {
        "cat"
    } else if is_torch_qualified_name(semantic, &call.func, "torch.stack") {
        "stack"
    } else {
        return;
    };

    // Walk outward from the current statement; flag if any enclosing statement
    // is a `for`/`while` loop *and* we haven't crossed a function/class
    // boundary first (those reset the loop context).
    for stmt in semantic.current_statements() {
        match stmt {
            Stmt::For(_) | Stmt::While(_) => {
                checker.report_diagnostic(CatStackInLoop { name }, call.func.range());
                return;
            }
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => return,
            _ => {}
        }
    }
}
