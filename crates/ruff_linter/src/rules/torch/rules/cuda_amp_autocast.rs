use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for uses of the deprecated `torch.cuda.amp.autocast` and suggests
/// the device-agnostic `torch.amp.autocast("cuda")` instead.
///
/// ## Why is this bad?
/// `torch.cuda.amp.autocast` has been deprecated in favor of the
/// device-agnostic `torch.amp.autocast` introduced in PyTorch 1.10. The new
/// entry point takes the device type as its first argument and works
/// identically for CPU, CUDA, and other backends.
///
/// ## Example
/// ```python
/// import torch
///
/// with torch.cuda.amp.autocast():
///     out = model(x)
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// with torch.amp.autocast("cuda"):
///     out = model(x)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because it inserts a `"cuda"` positional argument
/// that may shadow a later `device_type=` keyword the user had supplied via
/// `**kwargs`. In typical usage the rewrite is behaviour-preserving.
///
/// ## References
/// - [PyTorch documentation: `torch.amp.autocast`](https://pytorch.org/docs/stable/amp.html#torch.autocast)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct CudaAmpAutocast;

impl AlwaysFixableViolation for CudaAmpAutocast {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.cuda.amp.autocast` is deprecated; use `torch.amp.autocast(\"cuda\", ...)`"
            .to_string()
    }

    fn fix_title(&self) -> String {
        "Replace with `torch.amp.autocast(\"cuda\", ...)`".to_string()
    }
}

/// TORCH600
pub(crate) fn cuda_amp_autocast(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.cuda.amp.autocast") {
        return;
    }

    // Build a replacement for just the callee expression, leaving the existing
    // argument list intact. A `"cuda"` positional is prepended below.
    let callee_replacement = match call.func.as_ref() {
        Expr::Attribute(_) => "torch.amp.autocast".to_string(),
        Expr::Name(_) => "torch.amp.autocast".to_string(),
        _ => return,
    };

    let mut diagnostic = checker.report_diagnostic(CudaAmpAutocast, call.func.range());

    // Replace the callee.
    let callee_edit = Edit::range_replacement(callee_replacement, call.func.range());

    // Insert the `"cuda"` device argument at the front of the existing argument
    // list. `arguments.start()` points at the opening `(`, so insert directly
    // after it.
    let insert_pos = call.arguments.start() + ruff_text_size::TextSize::from(1);
    let has_existing_args = !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty();
    let device_arg = if has_existing_args {
        "\"cuda\", ".to_string()
    } else {
        "\"cuda\"".to_string()
    };
    let arg_edit = Edit::insertion(device_arg, insert_pos);

    diagnostic.set_fix(Fix::unsafe_edits(callee_edit, [arg_edit]));
}
