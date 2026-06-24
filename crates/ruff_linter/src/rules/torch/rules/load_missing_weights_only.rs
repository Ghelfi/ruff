use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::{Ranged, TextSize};

use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;
use crate::{AlwaysFixableViolation, Edit, Fix};

/// ## What it does
/// Checks for calls to `torch.load` that do not pass `weights_only=True`.
///
/// ## Why is this bad?
/// `torch.load` uses `pickle` under the hood and, in the default
/// `weights_only=False` mode, can execute arbitrary code embedded in a
/// checkpoint. Loading an untrusted file is equivalent to running an
/// untrusted Python script.
///
/// Starting in PyTorch 2.4, calling `torch.load` without specifying
/// `weights_only` raises a `FutureWarning`; the default will flip to
/// `True` in a future release. Set it explicitly to make the intent clear
/// and avoid breakage when the default changes.
///
/// ## Example
/// ```python
/// import torch
///
/// state_dict = torch.load("checkpoint.pt")
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// state_dict = torch.load("checkpoint.pt", weights_only=True)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because `weights_only=True` rejects checkpoints
/// that contain non-tensor objects (e.g., a pickled Python class or function).
/// Loading legitimate checkpoints with custom Python objects will start
/// raising after the fix is applied.
///
/// ## References
/// - [PyTorch documentation: `torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct LoadMissingWeightsOnly;

impl AlwaysFixableViolation for LoadMissingWeightsOnly {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.load` called without `weights_only=True`".to_string()
    }

    fn fix_title(&self) -> String {
        "Add `weights_only=True`".to_string()
    }
}

/// TORCH603
pub(crate) fn load_missing_weights_only(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.load") {
        return;
    }

    // `weights_only` is keyword-only on `torch.load`; pass `usize::MAX` so
    // the positional fallback can never spuriously match a different
    // argument.
    if call
        .arguments
        .find_argument_value("weights_only", usize::MAX)
        .is_some()
    {
        return;
    }

    let mut diagnostic = checker.report_diagnostic(LoadMissingWeightsOnly, call.func.range());

    // Insert the keyword just before the closing `)` of the call. The closing
    // paren is the last character of `call.arguments`, so insert at `end - 1`.
    let insert_pos = call.arguments.end() - TextSize::from(1);
    let has_existing_args = !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty();
    let insertion = if has_existing_args {
        ", weights_only=True".to_string()
    } else {
        "weights_only=True".to_string()
    };
    diagnostic.set_fix(Fix::unsafe_edit(Edit::insertion(insertion, insert_pos)));
}
