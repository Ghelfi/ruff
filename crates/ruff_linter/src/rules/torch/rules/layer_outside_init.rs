use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::{is_self_attribute, is_torch_module_subclass};

/// ## What it does
/// Checks for `nn.Module` subclasses that assign a `torch.nn.*` layer to a
/// `self` attribute outside `__init__` (typically in `forward` or another
/// helper method).
///
/// ## Why is this bad?
/// `nn.Module` registers submodules at *assignment time*. Assigning
/// `self.fc = nn.Linear(...)` inside `forward` reconstructs the layer on
/// every call, which:
///
/// - throws away the parameters learned by the previous step,
/// - never lets the optimizer see the new parameters (it was built from the
///   parameters that existed at `optimizer = Adam(model.parameters())` time),
/// - and skips serialization in `state_dict`.
///
/// Define all layers in `__init__` and only *use* them in `forward`.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         super().__init__()
///
///     def forward(self, x):
///         self.fc = nn.Linear(10, 10)  # Bad: rebuilt every call
///         return self.fc(x)
/// ```
///
/// Use instead:
/// ```python
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.fc = nn.Linear(10, 10)
///
///     def forward(self, x):
///         return self.fc(x)
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct LayerOutsideInit;

impl Violation for LayerOutsideInit {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Layer assigned to `self` outside `__init__`; define submodules in `__init__` instead"
            .to_string()
    }
}

/// TORCH202
pub(crate) fn layer_outside_init(checker: &Checker, assign: &ast::StmtAssign) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let [target] = assign.targets.as_slice() else {
        return;
    };
    if !is_self_attribute(target) {
        return;
    }

    if !is_nn_layer_call(semantic, &assign.value) {
        return;
    }

    if !in_non_init_method_of_nn_module(semantic) {
        return;
    }

    checker.report_diagnostic(LayerOutsideInit, assign.value.range());
}

/// Returns `true` if the enclosing function is a method on an `nn.Module`
/// subclass and is *not* `__init__`. Walks outward from the current
/// statement to find the innermost enclosing function and the class it
/// belongs to.
fn in_non_init_method_of_nn_module(semantic: &SemanticModel) -> bool {
    let mut statements = semantic.current_statements();
    let func = loop {
        match statements.next() {
            Some(Stmt::FunctionDef(func)) => break func,
            Some(_) => continue,
            None => return false,
        }
    };
    if func.name.as_str() == "__init__" {
        return false;
    }
    statements.any(|stmt| {
        matches!(stmt, Stmt::ClassDef(class_def) if is_torch_module_subclass(class_def, semantic))
    })
}

/// Returns `true` if `expr` is a call to a `torch.nn.*` module constructor
/// (e.g., `nn.Linear(...)`), excluding `nn.Parameter`.
fn is_nn_layer_call(semantic: &SemanticModel, expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    semantic
        .resolve_qualified_name(&call.func)
        .is_some_and(|qualified_name| match qualified_name.segments() {
            ["torch", "nn", name] => {
                name != &"Parameter" && name.starts_with(|c: char| c.is_ascii_uppercase())
            }
            _ => false,
        })
}
