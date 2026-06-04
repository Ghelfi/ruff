use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.tensor()` that do not specify a `device=`
/// keyword argument.
///
/// ## Why is this bad?
/// Without an explicit `device=`, the tensor is allocated on the default
/// device, which is almost always CPU. In code that runs on GPU, this leads
/// to a silent fallback to CPU or to a device-mismatch error the next time
/// the tensor is combined with a GPU tensor.
///
/// Being explicit about the device makes the placement obvious to readers
/// and avoids surprises when the same code runs on different hardware.
///
/// ## Example
/// ```python
/// import torch
///
/// # Bad: device is implicit (CPU by default)
/// x = torch.tensor([1.0, 2.0])
///
/// # Good: device is explicit
/// x = torch.tensor([1.0, 2.0], device="cuda")
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TensorMissingDevice;

impl Violation for TensorMissingDevice {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Missing explicit `device=` argument in `torch.tensor()` call".to_string()
    }
}

/// TORCH008
pub(crate) fn tensor_missing_device(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.tensor") {
        return;
    }

    if call.arguments.find_keyword("device").is_some() {
        return;
    }

    checker.report_diagnostic(TensorMissingDevice, call.range());
}
