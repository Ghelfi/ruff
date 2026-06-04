use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::in_compiled_function;

/// ## What it does
/// Checks for assignments to `self.<attr>` inside a function decorated with
/// `@torch.compile`.
///
/// ## Why is this bad?
/// `TorchDynamo` treats `nn.Module` attributes as static during tracing. When
/// the compiled function mutates `self.<attr>`, the new value is invisible
/// to subsequent invocations of the compiled graph (or, in some versions,
/// triggers a recompile). Either way the behavior diverges from eager
/// execution and the optimization is lost.
///
/// State that genuinely needs to mutate at runtime should be stored in a
/// `register_buffer` tensor, returned from the function, or kept outside
/// the compiled region.
///
/// The detection is limited to direct `self.<attr> = ...` assignments and
/// augmented assignments (e.g., `self.counter += 1`). Attribute access via
/// other receivers is not analyzed.
///
/// ## Example
/// ```python
/// import torch
///
///
/// class M(torch.nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.counter = 0
///
///     @torch.compile
///     def forward(self, x):
///         self.counter += 1  # Bad: mutates module state
///         return x + self.counter
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.compile` and module state](https://pytorch.org/docs/stable/torch.compiler_faq.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ModuleStateMutation;

impl Violation for ModuleStateMutation {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Mutation of `self` attribute inside `@torch.compile` function is non-static".to_string()
    }
}

/// TORCH104 — `self.attr = ...`
pub(crate) fn module_state_assign(checker: &Checker, assign: &ast::StmtAssign) {
    let semantic = checker.semantic();
    if !in_compiled_function(semantic) {
        return;
    }
    for target in &assign.targets {
        if let Some(range) = self_attr_range(target) {
            checker.report_diagnostic(ModuleStateMutation, range);
        }
    }
}

/// TORCH104 — `self.attr += ...` and similar.
pub(crate) fn module_state_aug_assign(checker: &Checker, aug: &ast::StmtAugAssign) {
    let semantic = checker.semantic();
    if !in_compiled_function(semantic) {
        return;
    }
    if let Some(range) = self_attr_range(&aug.target) {
        checker.report_diagnostic(ModuleStateMutation, range);
    }
}

/// TORCH104 — `self.attr: T = ...`.
pub(crate) fn module_state_ann_assign(checker: &Checker, ann: &ast::StmtAnnAssign) {
    let semantic = checker.semantic();
    if ann.value.is_none() {
        return;
    }
    if !in_compiled_function(semantic) {
        return;
    }
    if let Some(range) = self_attr_range(&ann.target) {
        checker.report_diagnostic(ModuleStateMutation, range);
    }
}

/// If `expr` is an attribute access on the name `self`, return its range.
fn self_attr_range(expr: &Expr) -> Option<ruff_text_size::TextRange> {
    let Expr::Attribute(attribute) = expr else {
        return None;
    };
    let Expr::Name(name) = attribute.value.as_ref() else {
        return None;
    };
    if name.id.as_str() != "self" {
        return None;
    }
    Some(attribute.range())
}
