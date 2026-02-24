//! Torch-specific lint rules.

pub(crate) mod helpers;
pub mod rules;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use test_case::test_case;

    use crate::assert_diagnostics;
    use crate::registry::Rule;
    use crate::settings::LinterSettings;
    use crate::test::test_path;

    #[test_case(Rule::TensorConstructor, Path::new("TORCH001.py"))]
    #[test_case(Rule::TensorConstructor, Path::new("TORCH001_no_import.py"))]
    #[test_case(Rule::TensorConstructor, Path::new("TORCH001_noqa_all.py"))]
    #[test_case(Rule::TensorConstructor, Path::new("TORCH001_noqa_code.py"))]
    #[test_case(Rule::TensorConstructor, Path::new("TORCH001_noqa_code_per_line.py"))]
    fn rules(rule_code: Rule, path: &Path) -> Result<()> {
        let snapshot = format!("{}", path.to_string_lossy());
        let diagnostics = test_path(
            Path::new("torch").join(path).as_path(),
            &LinterSettings::for_rule(rule_code),
        )?;
        assert_diagnostics!(snapshot, diagnostics);
        Ok(())
    }
}
