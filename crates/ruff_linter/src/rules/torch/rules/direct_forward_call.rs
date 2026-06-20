use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for direct calls to a module's `.forward()` method
/// (e.g., `model.forward(x)`).
///
/// ## Why is this bad?
/// Calling `model(x)` invokes `nn.Module.__call__`, which runs the registered
/// forward and backward hooks (used by features such as `register_forward_hook`,
/// AMP, and `torch.compile`) around `forward`. Calling `model.forward(x)`
/// directly bypasses those hooks, so behavior silently diverges from the
/// rest of the framework.
///
/// Always invoke the module as a callable: `model(x)`.
///
/// ## Example
/// ```python
/// output = model.forward(x)
/// ```
///
/// Use instead:
/// ```python
/// output = model(x)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DirectForwardCall;

impl Violation for DirectForwardCall {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Call the module directly (`module(...)`) instead of `module.forward(...)`".to_string()
    }
}

/// TORCH213
pub(crate) fn direct_forward_call(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };

    if attribute.attr.as_str() != "forward" {
        return;
    }

    // `super().forward(...)` is a legitimate way to delegate to a base class's
    // implementation when overriding `forward`.
    if matches!(attribute.value.as_ref(), Expr::Call(inner)
        if matches!(inner.func.as_ref(), Expr::Name(name) if name.id.as_str() == "super"))
    {
        return;
    }

    checker.report_diagnostic(DirectForwardCall, call.func.range());
}
