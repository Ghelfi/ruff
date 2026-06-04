use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_python_semantic::SemanticModel;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::in_compiled_function;

/// ## What it does
/// Checks for `if` statements inside a `@torch.compile` function whose
/// condition depends on the value of a tensor.
///
/// ## Why is this bad?
/// TorchDynamo cannot specialize a graph on values it has not yet computed.
/// Branching on a tensor's value forces a graph break, falling back to
/// eager execution and recompiling on each new branch combination.
///
/// Prefer using vectorized constructs such as `torch.where`, `torch.cond`,
/// or moving the conditional out of the compiled function so the dispatch
/// happens before tracing.
///
/// The detection is heuristic: a condition is considered tensor-dependent
/// when it references a local name whose binding was initialized from a
/// `torch.*` call.
///
/// ## Example
/// ```python
/// import torch
///
///
/// @torch.compile
/// def f(x):
///     y = torch.zeros_like(x)
///     if y.sum() > 0:  # Bad: branch depends on a tensor
///         return x
///     return x + 1
///
///
/// @torch.compile
/// def f(x):
///     y = torch.zeros_like(x)
///     return torch.where(y.sum() > 0, x, x + 1)  # Good
/// ```
///
/// ## References
/// - [PyTorch documentation: `torch.compile` graph breaks](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DataDependentIf;

impl Violation for DataDependentIf {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`if` on a tensor value inside `@torch.compile` function causes a graph break".to_string()
    }
}

/// TORCH102
pub(crate) fn data_dependent_if(checker: &Checker, if_stmt: &ast::StmtIf) {
    let semantic = checker.semantic();

    if !in_compiled_function(semantic) {
        return;
    }

    if !condition_uses_tensor(semantic, &if_stmt.test) {
        return;
    }

    checker.report_diagnostic(DataDependentIf, if_stmt.test.range());
}

/// Returns `true` if any name in `expr` resolves to a binding whose
/// initializer is a `torch.*` call.
fn condition_uses_tensor(semantic: &SemanticModel, expr: &Expr) -> bool {
    let mut visitor = TensorNameFinder {
        semantic,
        found: false,
    };
    visitor.visit_expr(expr);
    visitor.found
}

struct TensorNameFinder<'a, 'sem> {
    semantic: &'sem SemanticModel<'a>,
    found: bool,
}

impl<'a> Visitor<'a> for TensorNameFinder<'a, '_> {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if self.found {
            return;
        }
        if let Expr::Name(name) = expr
            && name_is_tensor(self.semantic, name)
        {
            self.found = true;
            return;
        }
        visitor::walk_expr(self, expr);
    }
}

fn name_is_tensor(semantic: &SemanticModel, name: &ast::ExprName) -> bool {
    let Some(binding_id) = semantic.resolve_name(name) else {
        return false;
    };
    let binding = semantic.binding(binding_id);
    let Some(source) = binding.source else {
        return false;
    };

    let init = match semantic.statement(source) {
        Stmt::Assign(ast::StmtAssign { value, .. }) => value.as_ref(),
        Stmt::AnnAssign(ast::StmtAnnAssign {
            value: Some(value), ..
        }) => value.as_ref(),
        _ => return false,
    };

    let Expr::Call(call) = init else {
        return false;
    };

    semantic
        .resolve_qualified_name(&call.func)
        .is_some_and(|qname| qname.segments().first() == Some(&"torch"))
}
