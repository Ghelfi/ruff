//! Helpers for Torch-specific lint rules.
//!
//! This module provides semantic analysis utilities to robustly determine
//! whether a given expression refers to a `torch` symbol, handling:
//! - Direct `torch.X` attribute access (e.g., `torch.Tensor`)
//! - `from torch import X` style imports (e.g., `from torch import Tensor`)
//! - Aliased imports (e.g., `import torch as th; th.Tensor`)
//! - NOT matching user-defined classes named `Tensor` or other shadowed names.

use ruff_python_ast::{self as ast, Stmt, helpers::map_callable};
use ruff_python_semantic::SemanticModel;

/// Check whether a given call expression's callable resolves to a specific
/// fully-qualified torch symbol (e.g., `torch.Tensor`, `torch.tensor`).
///
/// This function resolves through:
/// - `import torch; torch.Tensor(...)` → qualified name `torch.Tensor`
/// - `import torch as th; th.Tensor(...)` → qualified name `torch.Tensor`
/// - `from torch import Tensor; Tensor(...)` → qualified name `torch.Tensor`
///
/// It will NOT match if the binding comes from a non-torch module or from a
/// local class definition.
pub(crate) fn is_torch_qualified_name(
    semantic: &SemanticModel,
    expr: &ruff_python_ast::Expr,
    qualified_name: &str,
) -> bool {
    semantic
        .resolve_qualified_name(expr)
        .is_some_and(|qname| qname.to_string() == qualified_name)
}

/// Returns `true` if any enclosing function (innermost first) is decorated
/// with `@torch.compile` or `@torch.compile(...)`.
///
/// Walks up the current statement hierarchy, stopping at the first function
/// definition encountered. Both the bare-decorator and decorator-factory
/// forms are recognized.
pub(crate) fn in_compiled_function(semantic: &SemanticModel) -> bool {
    semantic.current_statements().any(|stmt| {
        let Stmt::FunctionDef(func) = stmt else {
            return false;
        };
        is_compiled_function(func, semantic)
    })
}

/// Returns `true` if the given function is decorated with `@torch.compile`
/// (or `@torch.compile(...)`).
pub(crate) fn is_compiled_function(func: &ast::StmtFunctionDef, semantic: &SemanticModel) -> bool {
    func.decorator_list.iter().any(|decorator| {
        is_torch_qualified_name(
            semantic,
            map_callable(&decorator.expression),
            "torch.compile",
        )
    })
}
