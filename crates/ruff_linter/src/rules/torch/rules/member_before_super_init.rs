use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::Modules;
use ruff_text_size::{Ranged, TextRange};

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::{is_super_init, is_torch_module_subclass};

/// ## What it does
/// Checks for assignments to instance attributes (`self.<attr> = ...`) that
/// occur before the `super().__init__()` call in an `nn.Module` subclass's
/// `__init__`.
///
/// ## Why is this bad?
/// `nn.Module.__init__` initializes the internal `_parameters`, `_buffers`,
/// and `_modules` registries. Assigning a submodule, parameter, or buffer to
/// `self` before that call runs raises `AttributeError: cannot assign module
/// before Module.__init__() call`, because the registries do not exist yet.
///
/// Always call `super().__init__()` first, before any attribute assignment.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         self.fc = nn.Linear(10, 10)  # AttributeError
///         super().__init__()
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
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct MemberBeforeSuperInit;

impl Violation for MemberBeforeSuperInit {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Instance attribute assigned before `super().__init__()` call".to_string()
    }
}

/// TORCH201
pub(crate) fn member_before_super_init(checker: &Checker, class_def: &ast::StmtClassDef) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    if !is_torch_module_subclass(class_def, semantic) {
        return;
    }

    let Some(init) = class_def.body.iter().find_map(|stmt| match stmt {
        Stmt::FunctionDef(func) if func.name.as_str() == "__init__" => Some(func),
        _ => None,
    }) else {
        return;
    };

    // Find the `super().__init__()` call among the top-level statements. If it
    // is never called, `TORCH200` is the relevant diagnostic, not this one.
    let Some(super_index) = init
        .body
        .iter()
        .position(|stmt| matches!(stmt, Stmt::Expr(expr) if is_super_init_expr(&expr.value)))
    else {
        return;
    };

    for stmt in &init.body[..super_index] {
        if let Some(range) = self_assignment_range(stmt) {
            checker.report_diagnostic(MemberBeforeSuperInit, range);
        }
    }
}

/// Returns `true` if `expr` is a `super().__init__(...)` call.
fn is_super_init_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Call(call) if is_super_init(&call.func))
}

/// If `stmt` assigns to a `self.<attr>` target, return the range of that
/// target.
fn self_assignment_range(stmt: &Stmt) -> Option<TextRange> {
    match stmt {
        Stmt::Assign(assign) => assign.targets.iter().find_map(self_attr_range),
        Stmt::AnnAssign(ast::StmtAnnAssign {
            target,
            value: Some(_),
            ..
        }) => self_attr_range(target),
        Stmt::AugAssign(aug) => self_attr_range(&aug.target),
        _ => None,
    }
}

/// If `expr` is an attribute access on the name `self`, return its range.
fn self_attr_range(expr: &Expr) -> Option<TextRange> {
    let Expr::Attribute(attribute) = expr else {
        return None;
    };
    let Expr::Name(name) = attribute.value.as_ref() else {
        return None;
    };
    (name.id.as_str() == "self").then(|| attribute.range())
}
