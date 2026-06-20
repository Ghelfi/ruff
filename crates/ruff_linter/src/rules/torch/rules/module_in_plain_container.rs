use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_self_attribute;

/// ## What it does
/// Checks for `nn.Module` instances stored in a plain `list` or `dict`
/// assigned to an instance attribute.
///
/// ## Why is this bad?
/// `nn.Module` only registers submodules that are assigned directly as
/// attributes or held in an `nn.ModuleList` / `nn.ModuleDict`. Modules placed
/// in a plain `list` or `dict` are invisible to `.parameters()`,
/// `.to(device)`, `.state_dict()`, and `.train()` / `.eval()`. Their
/// parameters never receive gradients and never move with the rest of the
/// model, which silently breaks training.
///
/// Use `nn.ModuleList` or `nn.ModuleDict` so the submodules are registered.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.layers = [nn.Linear(10, 10) for _ in range(3)]  # Bad
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
///         self.layers = nn.ModuleList(nn.Linear(10, 10) for _ in range(3))
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
/// - [PyTorch documentation: `torch.nn.ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ModuleInPlainContainer {
    container: &'static str,
    suggested: &'static str,
}

impl Violation for ModuleInPlainContainer {
    #[derive_message_formats]
    fn message(&self) -> String {
        let ModuleInPlainContainer {
            container,
            suggested,
        } = self;
        format!("Store submodules in `nn.{suggested}` instead of a plain `{container}`")
    }
}

/// TORCH204
pub(crate) fn module_in_plain_container(checker: &Checker, assign: &ast::StmtAssign) {
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

    let (container, suggested) = match assign.value.as_ref() {
        Expr::List(list) if list.iter().any(|elt| is_nn_module_call(semantic, elt)) => {
            ("list", "ModuleList")
        }
        Expr::ListComp(comp) if is_nn_module_call(semantic, &comp.elt) => ("list", "ModuleList"),
        Expr::Dict(dict)
            if dict
                .iter_values()
                .any(|value| is_nn_module_call(semantic, value)) =>
        {
            ("dict", "ModuleDict")
        }
        Expr::DictComp(comp) if is_nn_module_call(semantic, &comp.value) => ("dict", "ModuleDict"),
        _ => return,
    };

    checker.report_diagnostic(
        ModuleInPlainContainer {
            container,
            suggested,
        },
        assign.value.range(),
    );
}

/// Returns `true` if `expr` is a call to a `torch.nn.*` module constructor
/// (e.g., `nn.Linear(...)`), excluding `nn.Parameter`.
fn is_nn_module_call(semantic: &SemanticModel, expr: &Expr) -> bool {
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
