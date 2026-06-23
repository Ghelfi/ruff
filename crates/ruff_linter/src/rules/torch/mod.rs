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
    #[test_case(Rule::TensorDataAccess, Path::new("TORCH002.py"))]
    #[test_case(Rule::TensorDataAccess, Path::new("TORCH002_noqa_all.py"))]
    #[test_case(Rule::TensorDataAccess, Path::new("TORCH002_noqa_code.py"))]
    #[test_case(Rule::TensorDataAccess, Path::new("TORCH002_noqa_code_per_line.py"))]
    #[test_case(Rule::NumpyMissingForce, Path::new("TORCH003.py"))]
    #[test_case(Rule::NumpyMissingForce, Path::new("TORCH003_noqa_all.py"))]
    #[test_case(Rule::NumpyMissingForce, Path::new("TORCH003_noqa_code.py"))]
    #[test_case(Rule::NumpyMissingForce, Path::new("TORCH003_noqa_code_per_line.py"))]
    #[test_case(Rule::MissingDetach, Path::new("TORCH004.py"))]
    #[test_case(Rule::MissingDetach, Path::new("TORCH004_noqa_all.py"))]
    #[test_case(Rule::MissingDetach, Path::new("TORCH004_noqa_code.py"))]
    #[test_case(Rule::MissingDetach, Path::new("TORCH004_noqa_code_per_line.py"))]
    #[test_case(Rule::MissingEval, Path::new("TORCH005.py"))]
    #[test_case(Rule::MissingEval, Path::new("TORCH005_noqa_all.py"))]
    #[test_case(Rule::MissingEval, Path::new("TORCH005_noqa_code.py"))]
    #[test_case(Rule::MissingEval, Path::new("TORCH005_noqa_code_per_line.py"))]
    #[test_case(Rule::NoGradToInferenceMode, Path::new("TORCH006.py"))]
    #[test_case(Rule::NoGradToInferenceMode, Path::new("TORCH006_noqa_all.py"))]
    #[test_case(Rule::NoGradToInferenceMode, Path::new("TORCH006_noqa_code.py"))]
    #[test_case(
        Rule::NoGradToInferenceMode,
        Path::new("TORCH006_noqa_code_per_line.py")
    )]
    #[test_case(Rule::InplaceLeafGrad, Path::new("TORCH007.py"))]
    #[test_case(Rule::InplaceLeafGrad, Path::new("TORCH007_noqa_all.py"))]
    #[test_case(Rule::InplaceLeafGrad, Path::new("TORCH007_noqa_code.py"))]
    #[test_case(Rule::InplaceLeafGrad, Path::new("TORCH007_noqa_code_per_line.py"))]
    #[test_case(Rule::TensorMissingDevice, Path::new("TORCH008.py"))]
    #[test_case(Rule::TensorMissingDevice, Path::new("TORCH008_noqa_all.py"))]
    #[test_case(Rule::TensorMissingDevice, Path::new("TORCH008_noqa_code.py"))]
    #[test_case(Rule::TensorMissingDevice, Path::new("TORCH008_noqa_code_per_line.py"))]
    #[test_case(Rule::DeviceMismatch, Path::new("TORCH009.py"))]
    #[test_case(Rule::DeviceMismatch, Path::new("TORCH009_noqa_all.py"))]
    #[test_case(Rule::DeviceMismatch, Path::new("TORCH009_noqa_code.py"))]
    #[test_case(Rule::DeviceMismatch, Path::new("TORCH009_noqa_code_per_line.py"))]
    #[test_case(Rule::TensorCopyConstructor, Path::new("TORCH010.py"))]
    #[test_case(Rule::TensorCopyConstructor, Path::new("TORCH010_noqa_all.py"))]
    #[test_case(Rule::TensorCopyConstructor, Path::new("TORCH010_noqa_code.py"))]
    #[test_case(
        Rule::TensorCopyConstructor,
        Path::new("TORCH010_noqa_code_per_line.py")
    )]
    #[test_case(Rule::CloneWithoutDetach, Path::new("TORCH011.py"))]
    #[test_case(Rule::CloneWithoutDetach, Path::new("TORCH011_noqa_all.py"))]
    #[test_case(Rule::CloneWithoutDetach, Path::new("TORCH011_noqa_code.py"))]
    #[test_case(Rule::CloneWithoutDetach, Path::new("TORCH011_noqa_code_per_line.py"))]
    #[test_case(Rule::UseToMethod, Path::new("TORCH012.py"))]
    #[test_case(Rule::UseToMethod, Path::new("TORCH012_noqa_all.py"))]
    #[test_case(Rule::UseToMethod, Path::new("TORCH012_noqa_code.py"))]
    #[test_case(Rule::UseToMethod, Path::new("TORCH012_noqa_code_per_line.py"))]
    #[test_case(Rule::SqueezeWithoutDim, Path::new("TORCH013.py"))]
    #[test_case(Rule::SqueezeWithoutDim, Path::new("TORCH013_noqa_all.py"))]
    #[test_case(Rule::SqueezeWithoutDim, Path::new("TORCH013_noqa_code.py"))]
    #[test_case(Rule::SqueezeWithoutDim, Path::new("TORCH013_noqa_code_per_line.py"))]
    #[test_case(Rule::TensorThenTo, Path::new("TORCH014.py"))]
    #[test_case(Rule::TensorThenTo, Path::new("TORCH014_noqa_all.py"))]
    #[test_case(Rule::TensorThenTo, Path::new("TORCH014_noqa_code.py"))]
    #[test_case(Rule::TensorThenTo, Path::new("TORCH014_noqa_code_per_line.py"))]
    #[test_case(Rule::PrintInCompile, Path::new("TORCH100.py"))]
    #[test_case(Rule::PrintInCompile, Path::new("TORCH100_noqa_all.py"))]
    #[test_case(Rule::PrintInCompile, Path::new("TORCH100_noqa_code.py"))]
    #[test_case(Rule::PrintInCompile, Path::new("TORCH100_noqa_code_per_line.py"))]
    #[test_case(Rule::TryInCompile, Path::new("TORCH101.py"))]
    #[test_case(Rule::TryInCompile, Path::new("TORCH101_noqa_all.py"))]
    #[test_case(Rule::TryInCompile, Path::new("TORCH101_noqa_code.py"))]
    #[test_case(Rule::TryInCompile, Path::new("TORCH101_noqa_code_per_line.py"))]
    #[test_case(Rule::DataDependentIf, Path::new("TORCH102.py"))]
    #[test_case(Rule::DataDependentIf, Path::new("TORCH102_noqa_all.py"))]
    #[test_case(Rule::DataDependentIf, Path::new("TORCH102_noqa_code.py"))]
    #[test_case(Rule::DataDependentIf, Path::new("TORCH102_noqa_code_per_line.py"))]
    #[test_case(Rule::ItemInCompile, Path::new("TORCH103.py"))]
    #[test_case(Rule::ItemInCompile, Path::new("TORCH103_noqa_all.py"))]
    #[test_case(Rule::ItemInCompile, Path::new("TORCH103_noqa_code.py"))]
    #[test_case(Rule::ItemInCompile, Path::new("TORCH103_noqa_code_per_line.py"))]
    #[test_case(Rule::ModuleStateMutation, Path::new("TORCH104.py"))]
    #[test_case(Rule::ModuleStateMutation, Path::new("TORCH104_noqa_all.py"))]
    #[test_case(Rule::ModuleStateMutation, Path::new("TORCH104_noqa_code.py"))]
    #[test_case(Rule::ModuleStateMutation, Path::new("TORCH104_noqa_code_per_line.py"))]
    #[test_case(Rule::MissingSuperInit, Path::new("TORCH200.py"))]
    #[test_case(Rule::MissingSuperInit, Path::new("TORCH200_noqa_all.py"))]
    #[test_case(Rule::MissingSuperInit, Path::new("TORCH200_noqa_code.py"))]
    #[test_case(Rule::MissingSuperInit, Path::new("TORCH200_noqa_code_per_line.py"))]
    #[test_case(Rule::MemberBeforeSuperInit, Path::new("TORCH201.py"))]
    #[test_case(Rule::MemberBeforeSuperInit, Path::new("TORCH201_noqa_all.py"))]
    #[test_case(Rule::MemberBeforeSuperInit, Path::new("TORCH201_noqa_code.py"))]
    #[test_case(
        Rule::MemberBeforeSuperInit,
        Path::new("TORCH201_noqa_code_per_line.py")
    )]
    #[test_case(Rule::ModuleInPlainContainer, Path::new("TORCH204.py"))]
    #[test_case(Rule::ModuleInPlainContainer, Path::new("TORCH204_noqa_all.py"))]
    #[test_case(Rule::ModuleInPlainContainer, Path::new("TORCH204_noqa_code.py"))]
    #[test_case(
        Rule::ModuleInPlainContainer,
        Path::new("TORCH204_noqa_code_per_line.py")
    )]
    #[test_case(Rule::ParameterInPlainContainer, Path::new("TORCH210.py"))]
    #[test_case(Rule::ParameterInPlainContainer, Path::new("TORCH210_noqa_all.py"))]
    #[test_case(Rule::ParameterInPlainContainer, Path::new("TORCH210_noqa_code.py"))]
    #[test_case(
        Rule::ParameterInPlainContainer,
        Path::new("TORCH210_noqa_code_per_line.py")
    )]
    #[test_case(Rule::WeightNormDeprecated, Path::new("TORCH206.py"))]
    #[test_case(Rule::WeightNormDeprecated, Path::new("TORCH206_noqa_all.py"))]
    #[test_case(Rule::WeightNormDeprecated, Path::new("TORCH206_noqa_code.py"))]
    #[test_case(
        Rule::WeightNormDeprecated,
        Path::new("TORCH206_noqa_code_per_line.py")
    )]
    #[test_case(Rule::DeprecatedLinalg, Path::new("TORCH208.py"))]
    #[test_case(Rule::DeprecatedLinalg, Path::new("TORCH208_noqa_all.py"))]
    #[test_case(Rule::DeprecatedLinalg, Path::new("TORCH208_noqa_code.py"))]
    #[test_case(Rule::DeprecatedLinalg, Path::new("TORCH208_noqa_code_per_line.py"))]
    #[test_case(Rule::ArangeFloatStep, Path::new("TORCH207.py"))]
    #[test_case(Rule::ArangeFloatStep, Path::new("TORCH207_noqa_all.py"))]
    #[test_case(Rule::ArangeFloatStep, Path::new("TORCH207_noqa_code.py"))]
    #[test_case(Rule::ArangeFloatStep, Path::new("TORCH207_noqa_code_per_line.py"))]
    #[test_case(Rule::DataLoaderMissingWorkerInitFn, Path::new("TORCH303.py"))]
    #[test_case(Rule::DataLoaderMissingWorkerInitFn, Path::new("TORCH303_noqa_all.py"))]
    #[test_case(
        Rule::DataLoaderMissingWorkerInitFn,
        Path::new("TORCH303_noqa_code.py")
    )]
    #[test_case(
        Rule::DataLoaderMissingWorkerInitFn,
        Path::new("TORCH303_noqa_code_per_line.py")
    )]
    #[test_case(Rule::DataLoaderDistributedDropLast, Path::new("TORCH300.py"))]
    #[test_case(Rule::DataLoaderDistributedDropLast, Path::new("TORCH300_noqa_all.py"))]
    #[test_case(
        Rule::DataLoaderDistributedDropLast,
        Path::new("TORCH300_noqa_code.py")
    )]
    #[test_case(
        Rule::DataLoaderDistributedDropLast,
        Path::new("TORCH300_noqa_code_per_line.py")
    )]
    #[test_case(Rule::CatStackInLoop, Path::new("TORCH504.py"))]
    #[test_case(Rule::CatStackInLoop, Path::new("TORCH504_noqa_all.py"))]
    #[test_case(Rule::CatStackInLoop, Path::new("TORCH504_noqa_code.py"))]
    #[test_case(Rule::CatStackInLoop, Path::new("TORCH504_noqa_code_per_line.py"))]
    #[test_case(Rule::ZeroGradSetToNone, Path::new("TORCH501.py"))]
    #[test_case(Rule::ZeroGradSetToNone, Path::new("TORCH501_noqa_all.py"))]
    #[test_case(Rule::ZeroGradSetToNone, Path::new("TORCH501_noqa_code.py"))]
    #[test_case(Rule::ZeroGradSetToNone, Path::new("TORCH501_noqa_code_per_line.py"))]
    #[test_case(Rule::CudaAmpAutocast, Path::new("TORCH600.py"))]
    #[test_case(Rule::CudaAmpAutocast, Path::new("TORCH600_noqa_all.py"))]
    #[test_case(Rule::CudaAmpAutocast, Path::new("TORCH600_noqa_code.py"))]
    #[test_case(Rule::CudaAmpAutocast, Path::new("TORCH600_noqa_code_per_line.py"))]
    #[test_case(Rule::CudaAmpGradScaler, Path::new("TORCH601.py"))]
    #[test_case(Rule::CudaAmpGradScaler, Path::new("TORCH601_noqa_all.py"))]
    #[test_case(Rule::CudaAmpGradScaler, Path::new("TORCH601_noqa_code.py"))]
    #[test_case(Rule::CudaAmpGradScaler, Path::new("TORCH601_noqa_code_per_line.py"))]
    #[test_case(Rule::ClipGradValueDeprecated, Path::new("TORCH602.py"))]
    #[test_case(Rule::ClipGradValueDeprecated, Path::new("TORCH602_noqa_all.py"))]
    #[test_case(Rule::ClipGradValueDeprecated, Path::new("TORCH602_noqa_code.py"))]
    #[test_case(
        Rule::ClipGradValueDeprecated,
        Path::new("TORCH602_noqa_code_per_line.py")
    )]
    #[test_case(Rule::LoadMissingWeightsOnly, Path::new("TORCH603.py"))]
    #[test_case(Rule::LoadMissingWeightsOnly, Path::new("TORCH603_noqa_all.py"))]
    #[test_case(Rule::LoadMissingWeightsOnly, Path::new("TORCH603_noqa_code.py"))]
    #[test_case(
        Rule::LoadMissingWeightsOnly,
        Path::new("TORCH603_noqa_code_per_line.py")
    )]
    #[test_case(Rule::LoadMissingMapLocation, Path::new("TORCH604.py"))]
    #[test_case(Rule::LoadMissingMapLocation, Path::new("TORCH604_noqa_all.py"))]
    #[test_case(Rule::LoadMissingMapLocation, Path::new("TORCH604_noqa_code.py"))]
    #[test_case(
        Rule::LoadMissingMapLocation,
        Path::new("TORCH604_noqa_code_per_line.py")
    )]
    #[test_case(Rule::HubLoadTrustRepo, Path::new("TORCH605.py"))]
    #[test_case(Rule::HubLoadTrustRepo, Path::new("TORCH605_noqa_all.py"))]
    #[test_case(Rule::HubLoadTrustRepo, Path::new("TORCH605_noqa_code.py"))]
    #[test_case(Rule::HubLoadTrustRepo, Path::new("TORCH605_noqa_code_per_line.py"))]
    #[test_case(Rule::DirectForwardCall, Path::new("TORCH213.py"))]
    #[test_case(Rule::DirectForwardCall, Path::new("TORCH213_noqa_all.py"))]
    #[test_case(Rule::DirectForwardCall, Path::new("TORCH213_noqa_code.py"))]
    #[test_case(Rule::DirectForwardCall, Path::new("TORCH213_noqa_code_per_line.py"))]
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
