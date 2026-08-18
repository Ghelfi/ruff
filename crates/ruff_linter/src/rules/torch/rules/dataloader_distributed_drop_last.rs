use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;
use crate::rules::torch::helpers::{is_torch_qualified_name, resolve_single_step};

/// ## What it does
/// Checks for `DataLoader` instantiated with a `DistributedSampler` and no
/// `drop_last=True`.
///
/// ## Why is this bad?
/// `DistributedSampler` divides the dataset by world size; when the dataset
/// length is not evenly divisible, some ranks receive a shorter final batch
/// and other ranks pad it. Without `drop_last=True` on the `DataLoader`,
/// those uneven batches reach the model and cause subtle issues — for
/// example, `DistributedDataParallel` will hang or assert if the gradient
/// shapes differ across ranks because of an uneven last batch.
///
/// ## Example
/// ```python
/// from torch.utils.data import DataLoader, DistributedSampler
///
/// loader = DataLoader(dataset, sampler=DistributedSampler(dataset))
/// ```
///
/// Use instead:
/// ```python
/// from torch.utils.data import DataLoader, DistributedSampler
///
/// loader = DataLoader(
///     dataset, sampler=DistributedSampler(dataset), drop_last=True
/// )
/// ```
///
/// ## References
/// - [PyTorch documentation: `DistributedSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DataLoaderDistributedDropLast;

impl Violation for DataLoaderDistributedDropLast {
    #[derive_message_formats]
    fn message(&self) -> String {
        "`DataLoader` with `DistributedSampler` should set `drop_last=True` to avoid uneven last batches across ranks".to_string()
    }
}

/// TORCH300
pub(crate) fn dataloader_distributed_drop_last(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !is_torch_qualified_name(semantic, &call.func, "torch.utils.data.DataLoader") {
        return;
    }

    // `sampler` is the 4th positional parameter on `DataLoader.__init__`.
    // Follow a single-step `sampler = DistributedSampler(...); DataLoader(...
    // sampler=sampler)` binding so the common two-line form is caught
    // alongside the inline construction.
    let Some(sampler) = call.arguments.find_argument_value("sampler", 3) else {
        return;
    };

    if !is_distributed_sampler(semantic, resolve_single_step(semantic, sampler)) {
        return;
    }

    // If `drop_last` is already explicitly set, the user has made their
    // choice; trust them.
    if call
        .arguments
        .find_argument_value("drop_last", 12)
        .is_some()
    {
        return;
    }

    checker.report_diagnostic(DataLoaderDistributedDropLast, call.func.range());
}

/// Returns `true` if `expr` is a direct call to
/// `torch.utils.data.DistributedSampler` (or its `distributed` alias).
fn is_distributed_sampler(semantic: &ruff_python_semantic::SemanticModel, expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    semantic
        .resolve_qualified_name(&call.func)
        .is_some_and(|qualified_name| {
            matches!(
                qualified_name.segments(),
                ["torch", "utils", "data", "DistributedSampler"]
                    | [
                        "torch",
                        "utils",
                        "data",
                        "distributed",
                        "DistributedSampler"
                    ]
            )
        })
}
