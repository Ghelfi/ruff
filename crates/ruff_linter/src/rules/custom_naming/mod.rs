//! Rules from custom-naming.
pub(crate) mod rules;
pub mod settings;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use test_case::test_case;

    use crate::registry::Rule;
    use crate::test::test_path;
    use crate::{assert_messages, settings};

    #[test_case(Rule::InvalidTestName, Path::new("C001.py"))]
    fn rules(rule_code: Rule, path: &Path) -> Result<()> {
        let snapshot = format!("{}_{}", rule_code.noqa_code(), path.to_string_lossy());
        let diagnostics = test_path(
            Path::new("custom_naming").join(path).as_path(),
            &settings::LinterSettings {
                ..settings::LinterSettings::for_rule(rule_code)
            },
        )?;
        assert_messages!(snapshot, diagnostics);
        Ok(())
    }

    // #[test]
    // fn classmethod_decorators() -> Result<()> {
    //     let diagnostics = test_path(
    //         Path::new("custom_naming").join("C001.py").as_path(),
    //         &settings::LinterSettings {
    //             custom_naming: custom_naming::settings::Settings {
    //                 classmethod_decorators: vec![
    //                     "classmethod".to_string(),
    //                     "pydantic.validator".to_string(),
    //                     "expression".to_string(),
    //                 ],
    //                 ..Default::default()
    //             },
    //             ..settings::LinterSettings::for_rule(Rule::InvalidFirstArgumentNameForMethod)
    //         },
    //     )?;
    //     assert_messages!(diagnostics);
    //     Ok(())
    // }

    // #[test]
    // fn staticmethod_decorators() -> Result<()> {
    //     let diagnostics = test_path(
    //         Path::new("custom_naming").join("N805.py").as_path(),
    //         &settings::LinterSettings {
    //             custom_naming: custom_naming::settings::Settings {
    //                 staticmethod_decorators: vec![
    //                     "staticmethod".to_string(),
    //                     "thisisstatic".to_string(),
    //                 ],
    //                 ..Default::default()
    //             },
    //             ..settings::LinterSettings::for_rule(Rule::InvalidFirstArgumentNameForMethod)
    //         },
    //     )?;
    //     assert_messages!(diagnostics);
    //     Ok(())
    // }

    // #[test_case(Rule::InvalidTestName, Path::new("C001.py"))]
    // fn ignore_names(rule_code: Rule, path: &str) -> Result<()> {
    //     let snapshot = format!("ignore_names_{}_{path}", rule_code.noqa_code());
    //     let diagnostics = test_path(
    //         PathBuf::from_iter(["custom_naming", "ignore_names", path]).as_path(),
    //         &settings::LinterSettings {
    //             custom_naming: custom_naming::settings::Settings {
    //                 ignore_names: IgnoreNames::from_patterns([
    //                     "*allowed*".to_string(),
    //                     "*Allowed*".to_string(),
    //                     "*ALLOWED*".to_string(),
    //                     "BA".to_string(), // For N817.
    //                 ])
    //                 .unwrap(),
    //                 ..Default::default()
    //             },
    //             ..settings::LinterSettings::for_rule(rule_code)
    //         },
    //     )?;
    //     assert_messages!(snapshot, diagnostics);
    //     Ok(())
    // }
}
