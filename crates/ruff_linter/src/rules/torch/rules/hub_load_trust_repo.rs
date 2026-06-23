use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast as ast;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for calls to `torch.hub.load` that do not pass an explicit
/// `trust_repo` argument.
///
/// ## Why is this bad?
/// `torch.hub.load` downloads and executes arbitrary Python code from a Git
/// repository. Since PyTorch 1.12 the function requires the caller to opt in
/// by passing `trust_repo=` — leaving it unset triggers an interactive
/// prompt in scripts and a warning in CI, and a future PyTorch release plans
/// to make it raise.
///
/// Set `trust_repo=` explicitly: `True` to whitelist the repo (after
/// reviewing it), `False` to refuse, or `"check"` to defer to the
/// per-user trust cache.
///
/// ## Example
/// ```python
/// import torch
///
/// model = torch.hub.load("pytorch/vision", "resnet50")
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// model = torch.hub.load("pytorch/vision", "resnet50", trust_repo=True)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.hub.load`](https://pytorch.org/docs/stable/hub.html#torch.hub.load)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct HubLoadTrustRepo;

impl Violation for HubLoadTrustRepo {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.hub.load` called without `trust_repo=`; set it explicitly to silence the runtime prompt".to_string()
    }
}

/// TORCH605
pub(crate) fn hub_load_trust_repo(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.hub.load") {
        return;
    }

    // `trust_repo` is keyword-only; any positional index works for the lookup.
    if call
        .arguments
        .find_argument_value("trust_repo", usize::MAX)
        .is_some()
    {
        return;
    }

    checker.report_diagnostic(HubLoadTrustRepo, call.func.range());
}
