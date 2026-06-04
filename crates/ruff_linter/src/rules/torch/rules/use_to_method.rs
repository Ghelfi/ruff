use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_python_semantic::Modules;
use ruff_text_size::{Ranged, TextRange};

use crate::checkers::ast::Checker;
use crate::{Applicability, Edit, Fix, FixAvailability, Violation};

/// ## What it does
/// Checks for tensor method calls like `.cpu()`, `.cuda()`, `.float()`,
/// `.long()`, etc. and suggests using the general-purpose `.to()` method.
///
/// ## Why is this bad?
/// `.to(device)` and `.to(dtype)` are the canonical way to move a tensor or
/// change its dtype. The dedicated methods are convenient shortcuts but make
/// it harder to write device- or dtype-parametric code. They also predate
/// `.to()` and are mostly kept for backwards compatibility.
///
/// Using `.to()` consistently makes it easier to swap devices or dtypes via
/// a single variable.
///
/// ## Example
/// ```python
/// import torch
///
/// x = torch.tensor([1.0, 2.0])
///
/// # Bad
/// y = x.cuda()
/// z = x.float()
///
/// # Good
/// y = x.to("cuda")
/// z = x.to(torch.float32)
/// ```
///
/// ## Fix safety
/// The fix is unsafe because `.cuda()` and friends accept arguments that
/// don't map directly to `.to()` (e.g., `.cuda(non_blocking=True)`). When
/// the original call has extra arguments, the fix is omitted entirely.
///
/// ## References
/// - [PyTorch documentation: `Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct UseToMethod {
    method: String,
    suggested: String,
}

impl Violation for UseToMethod {
    const FIX_AVAILABILITY: FixAvailability = FixAvailability::Sometimes;

    #[derive_message_formats]
    fn message(&self) -> String {
        let UseToMethod { method, suggested } = self;
        format!("Use `.to({suggested})` instead of `.{method}()`")
    }

    fn fix_title(&self) -> Option<String> {
        let UseToMethod { method, suggested } = self;
        Some(format!("Replace `.{method}()` with `.to({suggested})`"))
    }
}

/// TORCH012
pub(crate) fn use_to_method(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    if !semantic.seen_module(Modules::TORCH) {
        return;
    }

    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };

    let method = attribute.attr.as_str();
    let Some(replacement) = to_replacement(method) else {
        return;
    };

    let mut diagnostic = checker.report_diagnostic(
        UseToMethod {
            method: method.to_string(),
            suggested: replacement.to_string(),
        },
        call.func.range(),
    );

    // Only offer an autofix when the original call takes no arguments —
    // otherwise we'd risk dropping `non_blocking=`, `device=`, etc.
    if call.arguments.args.is_empty() && call.arguments.keywords.is_empty() {
        // Replace `method()` (just the attr name and parentheses) with `to(REPL)`,
        // preserving the receiver and the surrounding expression context.
        let edit_range = TextRange::new(attribute.attr.start(), call.end());
        diagnostic.set_fix(Fix::applicable_edit(
            Edit::range_replacement(format!("to({replacement})"), edit_range),
            Applicability::Unsafe,
        ));
    }
}

fn to_replacement(method: &str) -> Option<&'static str> {
    Some(match method {
        "cpu" => "\"cpu\"",
        "cuda" => "\"cuda\"",
        "float" => "torch.float32",
        "double" => "torch.float64",
        "half" => "torch.float16",
        "long" => "torch.int64",
        "int" => "torch.int32",
        "short" => "torch.int16",
        "char" => "torch.int8",
        "byte" => "torch.uint8",
        "bool" => "torch.bool",
        _ => return None,
    })
}
