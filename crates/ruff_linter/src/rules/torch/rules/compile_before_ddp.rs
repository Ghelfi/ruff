use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::SemanticModel;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::{is_torch_qualified_name, resolve_single_step};

/// ## What it does
/// Checks for `DistributedDataParallel(torch.compile(model))` — i.e.,
/// wrapping a compiled model with `DistributedDataParallel`.
///
/// ## Why is this bad?
/// `DistributedDataParallel` (DDP) attaches gradient-sync hooks to the model
/// it wraps. When `torch.compile` runs first, those hooks are baked into the
/// compiled graph and Dynamo cannot reason about them: in the best case
/// compilation falls back to eager mode, and in the worst case the
/// all-reduce is silently skipped.
///
/// Compile the DDP wrapper, not the inner model:
/// `torch.compile(DistributedDataParallel(model))`.
///
/// ## Example
/// ```python
/// import torch
/// from torch.nn.parallel import DistributedDataParallel
///
/// compiled = DistributedDataParallel(torch.compile(model))
/// ```
///
/// Use instead:
/// ```python
/// import torch
/// from torch.nn.parallel import DistributedDataParallel
///
/// ddp = DistributedDataParallel(model)
/// compiled = torch.compile(ddp)
/// ```
///
/// ## References
/// - [PyTorch documentation: `DistributedDataParallel` and `torch.compile`](https://pytorch.org/docs/stable/notes/ddp.html#torchdynamo-ddpoptimizer)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct CompileBeforeDdp;

impl Violation for CompileBeforeDdp {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`torch.compile` should wrap `DistributedDataParallel`, not the other way around"
            .to_string()
    }
}

/// TORCH402
pub(crate) fn compile_before_ddp(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_ddp(semantic, &call.func) {
        return;
    }

    // First positional argument to DDP is the wrapped module. Follow a
    // single-step `compiled = torch.compile(model); DDP(compiled)` binding
    // so the typical two-line form is caught alongside the direct nesting.
    let Some(module) = call.arguments.find_argument_value("module", 0) else {
        return;
    };

    let Expr::Call(inner) = resolve_single_step(semantic, module) else {
        return;
    };

    if !is_torch_qualified_name(semantic, &inner.func, "torch.compile") {
        return;
    }

    checker.report_diagnostic(CompileBeforeDdp, module.range());
}

/// Returns `true` if `expr` resolves to `torch.nn.parallel.DistributedDataParallel`.
fn is_ddp(semantic: &SemanticModel, expr: &Expr) -> bool {
    semantic
        .resolve_qualified_name(expr)
        .is_some_and(|qualified_name| {
            matches!(
                qualified_name.segments(),
                ["torch", "nn", "parallel", "DistributedDataParallel"]
                    | [
                        "torch",
                        "nn",
                        "parallel",
                        "distributed",
                        "DistributedDataParallel"
                    ]
            )
        })
}
