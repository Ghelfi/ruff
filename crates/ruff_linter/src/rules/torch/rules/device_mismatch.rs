use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for binary operations whose operands are explicitly placed on
/// different devices (e.g., combining a CPU tensor and a CUDA tensor).
///
/// ## Why is this bad?
/// PyTorch raises a `RuntimeError` when an operation is invoked on tensors
/// that live on different devices. Detecting the mismatch statically catches
/// the bug at lint time, before any GPU is even available.
///
/// The check is intentionally narrow: only operands whose outermost call
/// makes the device explicit (`.cpu()`, `.cuda()`, or `.to("cpu" | "cuda")`)
/// are considered. Implicit placements are not analyzed.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0]).cuda()
/// y = torch.tensor([2.0]).cpu()
///
/// # Bad: explicit device mismatch
/// z = x.cuda() + y.cpu()
///
/// # Good: move both operands to the same device
/// z = x.cuda() + y.cuda()
/// ```
///
/// ## References
/// - [PyTorch documentation: `Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DeviceMismatch;

impl Violation for DeviceMismatch {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Operands are explicitly placed on different devices (CPU vs. CUDA)".to_string()
    }
}

/// TORCH009
pub(crate) fn device_mismatch(checker: &Checker, binop: &ast::ExprBinOp) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let (Some(left), Some(right)) = (device_of(&binop.left), device_of(&binop.right)) else {
        return;
    };

    if left == right {
        return;
    }

    checker.report_diagnostic(DeviceMismatch, binop.range());
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda,
}

fn device_of(expr: &Expr) -> Option<Device> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(ast::ExprAttribute { attr, .. }) = call.func.as_ref() else {
        return None;
    };

    match attr.as_str() {
        "cpu" => Some(Device::Cpu),
        "cuda" => Some(Device::Cuda),
        "to" => {
            // Inspect the first positional argument or `device=` keyword.
            let device_arg = call
                .arguments
                .args
                .first()
                .or_else(|| call.arguments.find_keyword("device").map(|kw| &kw.value));
            device_arg.and_then(device_from_string)
        }
        _ => None,
    }
}

fn device_from_string(expr: &Expr) -> Option<Device> {
    let Expr::StringLiteral(literal) = expr else {
        return None;
    };
    let value = literal.value.to_str();
    // Accept e.g. "cuda", "cuda:0", "cpu".
    let prefix = value.split(':').next()?;
    match prefix {
        "cpu" => Some(Device::Cpu),
        "cuda" => Some(Device::Cuda),
        _ => None,
    }
}
