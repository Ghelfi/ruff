use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for calls to `.zero_grad(set_to_none=False)`.
///
/// ## Why is this bad?
/// Since PyTorch 2.0, `optimizer.zero_grad()` defaults to `set_to_none=True`,
/// which is both faster and more memory-friendly: it drops the gradient
/// tensors entirely rather than overwriting them with zeros. Passing
/// `set_to_none=False` explicitly opts back into the slower legacy
/// behaviour, which is rarely intentional.
///
/// ## Example
/// ```python
/// optimizer.zero_grad(set_to_none=False)
/// ```
///
/// Use instead:
/// ```python
/// optimizer.zero_grad(set_to_none=True)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because flipping `set_to_none` changes the
/// downstream type of `.grad` from a zeroed tensor to `None`. Code that
/// reads `param.grad` between `zero_grad()` and `backward()` (for example,
/// a custom gradient inspection step) will start hitting `None` and raise.
///
/// ## References
/// - [PyTorch documentation: `torch.optim.Optimizer.zero_grad`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ZeroGradSetToNone;

impl AlwaysFixableViolation for ZeroGradSetToNone {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`zero_grad(set_to_none=False)` opts into the slower legacy behaviour; prefer the default `set_to_none=True`".to_string()
    }

    fn fix_title(&self) -> String {
        "Change `set_to_none=False` to `set_to_none=True`".to_string()
    }
}

/// TORCH501
pub(crate) fn zero_grad_set_to_none(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };
    if attribute.attr.as_str() != "zero_grad" {
        return;
    }

    let Some(keyword) = call.arguments.find_keyword("set_to_none") else {
        return;
    };

    let Expr::BooleanLiteral(ast::ExprBooleanLiteral { value: false, .. }) = &keyword.value else {
        return;
    };

    let mut diagnostic = checker.report_diagnostic(ZeroGradSetToNone, keyword.range());
    diagnostic.set_fix(Fix::unsafe_edit(Edit::range_replacement(
        "set_to_none=True".to_string(),
        keyword.range(),
    )));
}
