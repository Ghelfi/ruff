use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.load` that omit the `map_location` argument.
///
/// ## Why is this bad?
/// Without `map_location`, `torch.load` restores tensors to whichever device
/// they were saved on. A checkpoint trained on GPU 7 of one machine will try
/// to allocate GPU 7 on the loading machine, which silently fails (or worse,
/// loads onto an unintended device) when the topology differs.
///
/// Always pass `map_location` to make the target device explicit — typically
/// `"cpu"` for portability, or the actual device the model will run on.
///
/// ## Example
/// ```python
/// import torch
///
/// state_dict = torch.load("checkpoint.pt", weights_only=True)
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// state_dict = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct LoadMissingMapLocation;

impl Violation for LoadMissingMapLocation {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.load` called without `map_location`; tensors will be restored to their saved device"
            .to_string()
    }
}

/// TORCH604
pub(crate) fn load_missing_map_location(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.load") {
        return;
    }

    if call
        .arguments
        .find_argument_value("map_location", 1)
        .is_some()
    {
        return;
    }

    checker.report_diagnostic(LoadMissingMapLocation, call.func.range());
}
