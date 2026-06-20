use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::{is_super_init, is_torch_module_subclass};

/// ## What it does
/// Checks for `nn.Module` subclasses that define `__init__` but never call
/// `super().__init__()`.
///
/// ## Why is this bad?
/// `nn.Module.__init__` sets up the internal bookkeeping (`_parameters`,
/// `_buffers`, `_modules`, etc.) that every other method relies on. If a
/// subclass overrides `__init__` without calling `super().__init__()`, those
/// dictionaries are never created, and the first attempt to register a
/// submodule or parameter raises `AttributeError: cannot assign module before
/// Module.__init__() call`.
///
/// ## Example
/// ```python
/// import torch.nn as nn
///
///
/// class Net(nn.Module):
///     def __init__(self):
///         self.fc = nn.Linear(10, 10)  # AttributeError
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
pub(crate) struct MissingSuperInit;

impl Violation for MissingSuperInit {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`__init__` of `nn.Module` subclass does not call `super().__init__()`".to_string()
    }
}

/// TORCH200
pub(crate) fn missing_super_init(checker: &Checker, class_def: &ast::StmtClassDef) {
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

    if calls_super_init(&init.body) {
        return;
    }

    checker.report_diagnostic(MissingSuperInit, init.name.range());
}

/// Returns `true` if `body` contains a call to `super().__init__(...)`.
fn calls_super_init(body: &[Stmt]) -> bool {
    let mut visitor = SuperInitFinder { found: false };
    visitor.visit_body(body);
    visitor.found
}

struct SuperInitFinder {
    found: bool,
}

impl<'a> Visitor<'a> for SuperInitFinder {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if self.found {
            return;
        }
        if let Expr::Call(call) = expr
            && is_super_init(&call.func)
        {
            self.found = true;
            return;
        }
        visitor::walk_expr(self, expr);
    }
}
