use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_self_attribute;

/// ## What it does
/// Checks for `nn.Parameter` instances stored in a plain `list` or `dict`
/// assigned to an instance attribute.
///
/// ## Why is this bad?
/// `nn.Module` only registers parameters that are assigned directly as
/// attributes or held in an `nn.ParameterList` / `nn.ParameterDict`.
/// Parameters placed in a plain `list` or `dict` are invisible to
/// `.parameters()`, so the optimizer never updates them and they never move
/// with `.to(device)` — training silently does nothing for those weights.
///
/// Use `nn.ParameterList` or `nn.ParameterDict` so the parameters are
/// registered.
///
/// ## Example
/// ```python
/// import torch
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.weights = [nn.Parameter(torch.randn(10)) for _ in range(3)]  # Bad
/// ```
///
/// Use instead:
/// ```python
/// import torch
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.weights = nn.ParameterList(
///             nn.Parameter(torch.randn(10)) for _ in range(3)
///         )
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.ParameterList`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html)
/// - [PyTorch documentation: `torch.nn.ParameterDict`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct ParameterInPlainContainer {
    container: &'static str,
    suggested: &'static str,
}

impl Violation for ParameterInPlainContainer {
    #[derive_message_formats]
    fn message(&self) -> String {
        let ParameterInPlainContainer {
            container,
            suggested,
        } = self;
        format!("Store parameters in `nn.{suggested}` instead of a plain `{container}`")
    }
}

/// TORCH210
pub(crate) fn parameter_in_plain_container(checker: &Checker, assign: &ast::StmtAssign) {
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
        Expr::List(list) if list.iter().any(|elt| is_nn_parameter_call(semantic, elt)) => {
            ("list", "ParameterList")
        }
        Expr::ListComp(comp) if is_nn_parameter_call(semantic, &comp.elt) => {
            ("list", "ParameterList")
        }
        Expr::Dict(dict)
            if dict
                .iter_values()
                .any(|value| is_nn_parameter_call(semantic, value)) =>
        {
            ("dict", "ParameterDict")
        }
        Expr::DictComp(comp) if is_nn_parameter_call(semantic, &comp.value) => {
            ("dict", "ParameterDict")
        }
        _ => return,
    };

    checker.report_diagnostic(
        ParameterInPlainContainer {
            container,
            suggested,
        },
        assign.value.range(),
    );
}

/// Returns `true` if `expr` is a call to `torch.nn.Parameter(...)`.
fn is_nn_parameter_call(semantic: &SemanticModel, expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    semantic
        .resolve_qualified_name(&call.func)
        .is_some_and(|qualified_name| {
            matches!(
                qualified_name.segments(),
                ["torch", "nn", "Parameter"] | ["torch", "nn", "parameter", "Parameter"]
            )
        })
}
