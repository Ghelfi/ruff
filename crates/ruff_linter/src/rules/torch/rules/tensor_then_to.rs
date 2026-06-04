use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::{Ranged, TextRange};

use crate::checkers::ast::Checker;
use crate::fix::edits::add_argument;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{Applicability, Edit, Fix, FixAvailability, Violation};

/// ## What it does
/// Checks for `torch.tensor(...).to(device)` chains and suggests passing the
/// device directly to `torch.tensor()`.
///
/// ## Why is this bad?
/// `torch.tensor(...).to(device)` allocates the tensor on the CPU first, then
/// copies it to the target device. Passing `device=` to the constructor
/// skips the intermediate CPU allocation.
///
/// ## Example
/// ```python
/// import torch
///
/// # Bad: allocates on CPU, then moves to CUDA
/// x = torch.tensor([1.0]).to("cuda")
///
/// # Good: allocates directly on CUDA
/// x = torch.tensor([1.0], device="cuda")
/// ```
///
/// ## Fix safety
/// The fix is unsafe because `.to()` accepts forms that don't translate
/// directly to a `device=` keyword (e.g., `.to(dtype)`, `.to(other_tensor)`,
/// or `.to("cuda", non_blocking=True)`). The fix only applies when `.to()`
/// has a single positional device-like argument or a `device=` keyword.
///
/// ## References
/// - [PyTorch documentation: `torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html)
/// - [PyTorch documentation: `Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct TensorThenTo;

impl Violation for TensorThenTo {
    const FIX_AVAILABILITY: FixAvailability = FixAvailability::Sometimes;

    #[derive_message_formats]
    fn message(&self) -> String {
        "Avoid `torch.tensor(...).to(device)`; pass `device=` to `torch.tensor()` instead"
            .to_string()
    }

    fn fix_title(&self) -> Option<String> {
        Some("Move device into `torch.tensor()` call".to_string())
    }
}

/// TORCH014
pub(crate) fn tensor_then_to(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    // The outer call must be `<receiver>.to(...)`.
    let Expr::Attribute(ast::ExprAttribute { attr, value, .. }) = call.func.as_ref() else {
        return;
    };
    if attr.as_str() != "to" {
        return;
    }

    // The receiver must be a `torch.tensor(...)` call.
    let Expr::Call(inner_call) = value.as_ref() else {
        return;
    };
    if !is_torch_qualified_name(semantic, &inner_call.func, "torch.tensor") {
        return;
    }

    // Skip if `torch.tensor` already has a `device=` keyword (the `.to()` would be redundant
    // but rewriting it would create a duplicate keyword).
    if inner_call.arguments.find_keyword("device").is_some() {
        return;
    }

    // Skip `.to()` with no arguments — there's nothing to move.
    if call.arguments.args.is_empty() && call.arguments.keywords.is_empty() {
        return;
    }

    let mut diagnostic = checker.report_diagnostic(TensorThenTo, call.range());

    if let Some(device_text) = extract_device(checker, call) {
        let insert = add_argument(
            &format!("device={device_text}"),
            &inner_call.arguments,
            checker.tokens(),
        );
        let delete = Edit::range_deletion(TextRange::new(value.end(), call.end()));
        diagnostic.set_fix(Fix::applicable_edits(
            insert,
            [delete],
            Applicability::Unsafe,
        ));
    }
}

/// If `.to(...)` is called with a single device-like argument, return its
/// source text (suitable for inlining as a `device=` value).
fn extract_device(checker: &Checker, call: &ast::ExprCall) -> Option<String> {
    let locator = checker.locator();

    // Exactly one argument, positional or `device=`.
    if call.arguments.args.len() == 1 && call.arguments.keywords.is_empty() {
        return Some(locator.slice(call.arguments.args[0].range()).to_string());
    }
    if call.arguments.args.is_empty() && call.arguments.keywords.len() == 1 {
        let kw = &call.arguments.keywords[0];
        if kw.arg.as_ref().map(ast::Identifier::as_str) == Some("device") {
            return Some(locator.slice(kw.value.range()).to_string());
        }
    }
    None
}
