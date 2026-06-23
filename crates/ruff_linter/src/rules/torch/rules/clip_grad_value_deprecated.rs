use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.nn.utils.clip_grad_value_`.
///
/// ## Why is this bad?
/// Element-wise gradient clipping with `clip_grad_value_` discards directional
/// information from the gradient, which can hurt convergence. PyTorch
/// recommends norm-based clipping via `torch.nn.utils.clip_grad_norm_`,
/// which preserves the gradient direction while bounding its magnitude.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
/// nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
/// ```
///
/// Use instead:
/// ```python
/// import torch.nn as nn
///
/// nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.utils.clip_grad_norm_`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ClipGradValueDeprecated;

impl Violation for ClipGradValueDeprecated {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.nn.utils.clip_grad_value_` is discouraged; prefer `torch.nn.utils.clip_grad_norm_`"
            .to_string()
    }
}

/// TORCH602
pub(crate) fn clip_grad_value_deprecated(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.nn.utils.clip_grad_value_") {
        return;
    }

    checker.report_diagnostic(ClipGradValueDeprecated, call.func.range());
}
