use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr, Number, UnaryOp};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::is_torch_qualified_name;

/// ## What it does
/// Checks for `DataLoader(num_workers=N)` with `N > 0` and no
/// `worker_init_fn` set.
///
/// ## Why is this bad?
/// When `num_workers > 0`, each worker process inherits the parent's random
/// number generator state at fork time and never re-seeds it. Workers that
/// produce random augmentations (e.g., crops, flips, noise) will emit
/// identical sequences across epochs — and across workers — which silently
/// reduces the effective amount of augmentation and harms generalisation.
///
/// Provide a `worker_init_fn` that re-seeds `numpy`, Python `random`, and
/// any other PRNGs your dataset uses.
///
/// ## Example
/// ```python
/// from torch.utils.data import DataLoader
///
/// loader = DataLoader(dataset, num_workers=4)
/// ```
///
/// Use instead:
/// ```python
/// import numpy as np
/// import torch
/// from torch.utils.data import DataLoader
///
///
/// def seed_worker(worker_id):
///     seed = torch.initial_seed() % 2**32
///     np.random.seed(seed)
///
///
/// loader = DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)
/// ```
///
/// ## References
/// - [PyTorch documentation: `Reproducibility — DataLoader`](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DataLoaderMissingWorkerInitFn;

impl Violation for DataLoaderMissingWorkerInitFn {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`DataLoader(num_workers > 0)` without `worker_init_fn` leaves worker RNGs unseeded; data augmentation can repeat across epochs".to_string()
    }
}

/// TORCH303
pub(crate) fn dataloader_missing_worker_init_fn(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.utils.data.DataLoader") {
        return;
    }

    // `num_workers` is the 6th positional parameter on `DataLoader.__init__`.
    let Some(num_workers) = call.arguments.find_argument_value("num_workers", 5) else {
        return;
    };

    if !is_positive_int_literal(num_workers) {
        return;
    }

    if call
        .arguments
        .find_argument_value("worker_init_fn", 7)
        .is_some()
    {
        return;
    }

    checker.report_diagnostic(DataLoaderMissingWorkerInitFn, call.func.range());
}

/// Returns `true` if `expr` is an integer literal greater than zero,
/// transparently unwrapping a leading unary `+`.
fn is_positive_int_literal(expr: &Expr) -> bool {
    let expr = match expr {
        Expr::UnaryOp(ast::ExprUnaryOp {
            op: UnaryOp::UAdd,
            operand,
            ..
        }) => operand.as_ref(),
        _ => expr,
    };
    matches!(
        expr,
        Expr::NumberLiteral(ast::ExprNumberLiteral {
            value: Number::Int(value),
            ..
        }) if value.as_u64().is_some_and(|v| v > 0)
    )
}
