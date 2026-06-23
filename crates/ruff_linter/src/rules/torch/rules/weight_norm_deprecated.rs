use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for uses of the deprecated `torch.nn.utils.weight_norm` and suggests
/// `torch.nn.utils.parametrizations.weight_norm` instead.
///
/// ## Why is this bad?
/// `torch.nn.utils.weight_norm` was deprecated in PyTorch 2.1 in favor of the
/// new parametrization-based implementation at
/// `torch.nn.utils.parametrizations.weight_norm`. The new API integrates with
/// PyTorch's parametrization machinery, plays nicely with `state_dict`
/// serialization, and avoids subtle bugs around `remove_weight_norm`.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
/// linear = nn.utils.weight_norm(nn.Linear(10, 10))
/// ```
///
/// Use instead:
/// ```python
/// import torch.nn as nn
///
/// linear = nn.utils.parametrizations.weight_norm(nn.Linear(10, 10))
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because the new parametrization-based
/// `weight_norm` produces a `state_dict` with a slightly different layout
/// than the legacy implementation, so loading a checkpoint saved with the
/// legacy API may require an explicit migration step.
///
/// ## References
/// - [PyTorch documentation: `torch.nn.utils.parametrizations.weight_norm`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct WeightNormDeprecated;

impl AlwaysFixableViolation for WeightNormDeprecated {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.nn.utils.weight_norm` is deprecated; use `torch.nn.utils.parametrizations.weight_norm`".to_string()
    }

    fn fix_title(&self) -> String {
        "Replace with `torch.nn.utils.parametrizations.weight_norm`".to_string()
    }
}

/// TORCH206
pub(crate) fn weight_norm_deprecated(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.nn.utils.weight_norm") {
        return;
    }

    let replacement = match call.func.as_ref() {
        // Attribute form: `torch.nn.utils.weight_norm` or `nn.utils.weight_norm`.
        // Replace just the trailing `weight_norm` with `parametrizations.weight_norm`,
        // preserving the receiver expression as written.
        Expr::Attribute(ast::ExprAttribute { value, attr, .. }) => {
            let receiver = checker.locator().slice(value.range());
            Some((
                format!("{receiver}.parametrizations.{attr}"),
                call.func.range(),
            ))
        }
        // `from torch.nn.utils import weight_norm; weight_norm(...)` — no
        // unambiguous receiver to preserve, so we fall back to a fully
        // qualified replacement.
        Expr::Name(_) => Some((
            "torch.nn.utils.parametrizations.weight_norm".to_string(),
            call.func.range(),
        )),
        _ => None,
    };

    let mut diagnostic = checker.report_diagnostic(WeightNormDeprecated, call.func.range());
    if let Some((replacement, range)) = replacement {
        diagnostic.set_fix(Fix::unsafe_edit(Edit::range_replacement(
            replacement,
            range,
        )));
    }
}
