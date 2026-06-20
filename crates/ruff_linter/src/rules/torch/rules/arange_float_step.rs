use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Number, UnaryOp};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.arange` with a floating-point `step`.
///
/// ## Why is this bad?
/// With a floating-point `step`, the number of elements is computed as
/// `ceil((end - start) / step)`, which is sensitive to floating-point
/// rounding. A step such as `0.1` cannot be represented exactly, so the
/// resulting tensor can contain one more or one fewer element than expected,
/// silently changing tensor shapes downstream.
///
/// Prefer `torch.linspace`, which takes an explicit element count, when you
/// need evenly spaced floating-point values.
///
/// ## Example
/// ```python
/// import torch
///
/// torch.arange(0, 1, 0.1)  # May yield 10 or 11 elements
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// torch.linspace(0, 1, 11)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.arange`](https://pytorch.org/docs/stable/generated/torch.arange.html)
/// - [PyTorch documentation: `torch.linspace`](https://pytorch.org/docs/stable/generated/torch.linspace.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ArangeFloatStep;

impl Violation for ArangeFloatStep {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.arange` with a float `step` may produce an inconsistent element count".to_string()
    }
}

/// TORCH207
pub(crate) fn arange_float_step(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.arange") {
        return;
    }

    let Some(step) = call.arguments.find_argument_value("step", 2) else {
        return;
    };

    if is_float_literal(step) {
        checker.report_diagnostic(ArangeFloatStep, step.range());
    }
}

/// Returns `true` if `expr` is a float literal, ignoring a leading unary sign.
fn is_float_literal(expr: &Expr) -> bool {
    let expr = match expr {
        Expr::UnaryOp(ast::ExprUnaryOp {
            op: UnaryOp::USub | UnaryOp::UAdd,
            operand,
            ..
        }) => operand.as_ref(),
        _ => expr,
    };
    matches!(
        expr,
        Expr::NumberLiteral(ast::ExprNumberLiteral {
            value: Number::Float(_),
            ..
        })
    )
}
