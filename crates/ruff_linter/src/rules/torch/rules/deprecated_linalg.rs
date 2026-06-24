use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast, Expr};
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::{Applicability, Edit, Fix, FixAvailability, Violation};

/// ## What it does
/// Checks for calls to legacy `torch.*` linear-algebra functions that have
/// been superseded by equivalents in `torch.linalg`.
///
/// ## Why is this bad?
/// PyTorch's `torch.linalg` namespace provides a NumPy-compatible,
/// numerically stable replacement for the older free-standing linear-algebra
/// helpers under `torch`. A few of those legacy entry points (`torch.symeig`,
/// `torch.eig`, `torch.solve`, `torch.lstsq`) have been removed in recent
/// PyTorch releases; the others still exist but the `torch.linalg.*`
/// counterpart is the recommended, future-proof spelling.
///
/// Prefer the matching `torch.linalg.*` API.
///
/// ## Example
/// ```python
/// import torch
///
/// q, r = torch.qr(x)
/// ```
///
/// Use instead:
/// ```python
/// import torch
///
/// q, r = torch.linalg.qr(x)
/// ```
///
/// ## Fix safety
/// The fix is marked unsafe because some `torch.linalg.*` replacements have
/// slightly different signatures or return-value layouts than the legacy
/// `torch.*` versions (for example, `torch.linalg.solve` swaps the order of
/// arguments compared to `torch.solve`). Review the call site after applying.
///
/// ## References
/// - [PyTorch documentation: `torch.linalg`](https://pytorch.org/docs/stable/linalg.html)
#[derive(ViolationMetadata)]
#[violation_metadata(preview_since = "0.15.2")]
pub(crate) struct DeprecatedLinalg {
    name: &'static str,
    replacement: &'static str,
    status: LegacyStatus,
}

/// Whether the legacy `torch.<name>` entry point has been removed from
/// recent PyTorch releases, or merely superseded by a preferred
/// `torch.linalg.*` spelling.
#[derive(Copy, Clone, Eq, PartialEq)]
enum LegacyStatus {
    Removed,
    Superseded,
}

impl Violation for DeprecatedLinalg {
    const FIX_AVAILABILITY: FixAvailability = FixAvailability::Sometimes;

    #[derive_message_formats]
    fn message(&self) -> String {
        let DeprecatedLinalg {
            name,
            replacement,
            status,
        } = self;
        match status {
            LegacyStatus::Removed => format!(
                "`torch.{name}` has been removed in recent PyTorch; use `torch.linalg.{replacement}`"
            ),
            LegacyStatus::Superseded => {
                format!("`torch.{name}` is superseded by `torch.linalg.{replacement}`")
            }
        }
    }

    fn fix_title(&self) -> Option<String> {
        let DeprecatedLinalg {
            name, replacement, ..
        } = self;
        Some(format!(
            "Replace `torch.{name}` with `torch.linalg.{replacement}`"
        ))
    }
}

/// TORCH208
pub(crate) fn deprecated_linalg(checker: &Checker, call: &ast::ExprCall) {
    let semantic = checker.semantic();

    // Resolve the call's fully-qualified name once, then check the leaf.
    let Some(qualified) = semantic.resolve_qualified_name(&call.func) else {
        return;
    };
    let segments = qualified.segments();
    let &["torch", leaf] = segments else {
        return;
    };
    let Some((name, replacement, status)) = linalg_replacement(leaf) else {
        return;
    };

    // Only offer the autofix when the call is written as `<receiver>.<name>`
    // and the replacement is signature-compatible.
    let fix = match call.func.as_ref() {
        Expr::Attribute(ast::ExprAttribute { value, .. }) if is_signature_compatible(name) => {
            let receiver = checker.locator().slice(value.range());
            Some(Edit::range_replacement(
                format!("{receiver}.linalg.{replacement}"),
                call.func.range(),
            ))
        }
        Expr::Name(_) if is_signature_compatible(name) => Some(Edit::range_replacement(
            format!("torch.linalg.{replacement}"),
            call.func.range(),
        )),
        _ => None,
    };

    let mut diagnostic = checker.report_diagnostic(
        DeprecatedLinalg {
            name,
            replacement,
            status,
        },
        call.func.range(),
    );
    if let Some(edit) = fix {
        diagnostic.set_fix(Fix::applicable_edit(edit, Applicability::Unsafe));
    }
}

/// Map a legacy `torch.<name>` linalg function to its replacement under
/// `torch.linalg`, paired with whether the legacy entry point has been
/// removed or merely superseded.
fn linalg_replacement(name: &str) -> Option<(&'static str, &'static str, LegacyStatus)> {
    Some(match name {
        // Removed from recent PyTorch releases.
        "symeig" => ("symeig", "eigh", LegacyStatus::Removed),
        "eig" => ("eig", "eig", LegacyStatus::Removed),
        "solve" => ("solve", "solve", LegacyStatus::Removed),
        "lstsq" => ("lstsq", "lstsq", LegacyStatus::Removed),
        // Still present but superseded by the `torch.linalg` spelling.
        "qr" => ("qr", "qr", LegacyStatus::Superseded),
        "cholesky" => ("cholesky", "cholesky", LegacyStatus::Superseded),
        "matrix_rank" => ("matrix_rank", "matrix_rank", LegacyStatus::Superseded),
        "matrix_power" => ("matrix_power", "matrix_power", LegacyStatus::Superseded),
        "matrix_exp" => ("matrix_exp", "matrix_exp", LegacyStatus::Superseded),
        "pinverse" => ("pinverse", "pinv", LegacyStatus::Superseded),
        "inverse" => ("inverse", "inv", LegacyStatus::Superseded),
        "det" => ("det", "det", LegacyStatus::Superseded),
        "slogdet" => ("slogdet", "slogdet", LegacyStatus::Superseded),
        _ => return None,
    })
}

/// Returns `true` if the legacy `torch.<name>` and `torch.linalg.<replacement>`
/// take the same positional arguments in the same order and return values in
/// the same shape, making a plain rename safe.
///
/// Conservatively excludes `symeig`, `solve`, and `lstsq`, whose `torch.linalg`
/// counterparts have different argument orders or return tuples.
fn is_signature_compatible(name: &str) -> bool {
    !matches!(name, "symeig" | "solve" | "lstsq" | "eig")
}
