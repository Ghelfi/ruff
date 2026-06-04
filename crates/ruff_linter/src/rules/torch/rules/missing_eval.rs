use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::{Modules, SemanticModel};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for `with torch.no_grad():` or `with torch.inference_mode():`
/// blocks inside a function where no `.eval()` method call appears anywhere
/// in the enclosing function body.
///
/// ## Why is this bad?
/// Running inference on a model that is still in training mode produces
/// incorrect results: `Dropout` layers continue to drop activations and
/// `BatchNorm` layers update their running statistics. The
/// `with torch.no_grad():` context manager only disables autograd; it does
/// not switch the model out of training mode.
///
/// You should call `model.eval()` before running inference, and (optionally)
/// `model.train()` afterwards if training resumes.
///
/// Only inference blocks inside a function body are checked. Module-level
/// inference blocks are not flagged because the surrounding scope is too
/// broad to scan reliably without producing false positives.
///
/// ## Example
/// ```python
/// import torch
///
/// def predict(model, x):
///     # Bad: model is still in training mode
///     with torch.no_grad():
///         return model(x)
///
/// def predict(model, x):
///     # Good: switch to eval mode first
///     model.eval()
///     with torch.no_grad():
///         return model(x)
/// ```
///
/// ## References
/// - [PyTorch documentation: `Module.eval`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
/// - [PyTorch documentation: `torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct MissingEval;

impl Violation for MissingEval {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Missing `model.eval()` before inference block".to_string()
    }
}

/// TORCH005
pub(crate) fn missing_eval(checker: &Checker, with_stmt: &ast::StmtWith) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    // The with-statement must use `torch.no_grad()` or `torch.inference_mode()`
    // as a context manager.
    let triggers = with_stmt.items.iter().any(|item| {
        let Expr::Call(call) = &item.context_expr else {
            return false;
        };
        is_torch_qualified_name(semantic, &call.func, "torch.no_grad")
            || is_torch_qualified_name(semantic, &call.func, "torch.inference_mode")
    });
    if !triggers {
        return;
    }

    // Find the enclosing function. Skip the rule entirely at module level.
    let Some(function) = enclosing_function(semantic) else {
        return;
    };

    if scope_contains_eval_call(&function.body) {
        return;
    }

    checker.report_diagnostic(MissingEval, with_stmt.range());
}

/// Returns the nearest enclosing function definition, or `None` at module level.
fn enclosing_function<'a>(semantic: &SemanticModel<'a>) -> Option<&'a ast::StmtFunctionDef> {
    semantic.current_statements().find_map(|stmt| match stmt {
        Stmt::FunctionDef(func) => Some(func),
        _ => None,
    })
}

/// Returns `true` if any statement in `body` contains an `.eval()` method call.
/// Skips nested function and class bodies to keep the scan local.
fn scope_contains_eval_call(body: &[Stmt]) -> bool {
    let mut visitor = EvalCallFinder::default();
    for stmt in body {
        visitor.visit_stmt(stmt);
        if visitor.found {
            return true;
        }
    }
    visitor.found
}

#[derive(Default)]
struct EvalCallFinder {
    found: bool,
}

impl Visitor<'_> for EvalCallFinder {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.found {
            return;
        }
        // Don't descend into nested functions or classes — `.eval()` there
        // does not protect the enclosing scope.
        match stmt {
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
            _ => visitor::walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if self.found {
            return;
        }
        if let Expr::Call(call) = expr
            && let Expr::Attribute(ast::ExprAttribute { attr, .. }) = call.func.as_ref()
            && attr.as_str() == "eval"
        {
            self.found = true;
            return;
        }
        visitor::walk_expr(self, expr);
    }
}
