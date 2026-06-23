use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for uses of the deprecated `torch.cuda.amp.GradScaler` and suggests
/// the device-agnostic `torch.amp.GradScaler("cuda")` instead.
///
/// ## Why is this bad?
/// `torch.cuda.amp.GradScaler` has been deprecated in favor of the
/// device-agnostic `torch.amp.GradScaler`. The new constructor takes the
/// device type as its first argument and works identically across backends.
///
/// ## Example
/// ```python
/// import torch
///
/// scaler = torch.cuda.amp.GradScaler()
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// scaler = torch.amp.GradScaler("cuda")
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because it inserts a `"cuda"` positional argument
/// that may collide with later `device=` keyword passed dynamically (e.g.,
/// via `**kwargs`).
///
/// ## References
/// - [PyTorch documentation: `torch.amp.GradScaler`](https://pytorch.org/docs/stable/amp.html#torch.amp.GradScaler)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct CudaAmpGradScaler;

impl AlwaysFixableViolation for CudaAmpGradScaler {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.cuda.amp.GradScaler` is deprecated; use `torch.amp.GradScaler(\"cuda\", ...)`"
            .to_string()
    }

    fn fix_title(&self) -> String {
        "Replace with `torch.amp.GradScaler(\"cuda\", ...)`".to_string()
    }
}

/// TORCH601
pub(crate) fn cuda_amp_grad_scaler(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.cuda.amp.GradScaler") {
        return;
    }

    let callee_replacement = match call.func.as_ref() {
        Expr::Attribute(_) | Expr::Name(_) => "torch.amp.GradScaler".to_string(),
        _ => return,
    };

    let mut diagnostic = checker.report_diagnostic(CudaAmpGradScaler, call.func.range());

    let callee_edit = Edit::range_replacement(callee_replacement, call.func.range());
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
