# Codex Implementation Prompt: Funds Combined Uplift Model

Work in the current `prism_repo` repository and implement this task completely. Do not stop after proposing a plan.

## Objective

Create a duplicate of:

`Code/Uplift Model Code_rh06032026.ipynb`

The duplicate must perform the same uplift-modeling workflow, analyses, diagnostics, tables, charts, hyperparameter tuning, model comparisons, SHAP analyses, decile reporting, calibration, ROI analysis, and output generation, but it must use the Funds Combined dataset instead of the PRP dataset.

Create the new notebook as:

`Code/Uplift Model Code_Funds_Combined.ipynb`

Do not modify or overwrite the original notebook.

## Input data

Use `Funds_combined.csv`. Locate it in the repository before editing. It is currently expected at:

`prism_repo/DataSets/Funds_combined.csv`

There is currently a nested `prism_repo/` directory in the working tree, so verify the actual path rather than assuming the project root. If appropriate, place a copy of the dataset under the primary repository's `DataSets/` directory, but preserve the original file.

Read the file as CSV, not Excel. Normalize column names using the same `clean_names_simple` convention used by the source notebook.

## Required modeling definitions

- Treatment: derive the modeling treatment indicator from the combination of `intervention_flag` and `OptOut_flag`.
- Outcome: `outcome_ed_90d`
- Retain both raw treatment-source variables for derivation, auditing, and scored outputs. Do not discard `OptOut_flag` during preprocessing.
- Assign treated when the source `intervention_flag` equals 1 and does not conflict with an opt-out indication.
- Assign control when `OptOut_flag` equals 1 and does not conflict with a treated indication. This includes rows where the source `intervention_flag` is missing.
- Explicitly identify contradictory rows where both flags indicate treatment and opt-out, and unresolved rows where neither flag determines status. Report these counts before excluding any such rows.
- Store the original intervention value separately (for example, `source_intervention_flag`) and use a separately derived binary `intervention_flag` for modeling.
- The two raw source flags must be retained in audits and scored outputs but excluded from the predictor matrix because they directly define treatment status.
- Convert treatment to validated binary numeric values.
- Convert `outcome_ed_90d` to a binary ED-event indicator after replacing missing values with `0`: `0` means no ED visit and any positive ED-visit count becomes `1`. Do not discard members with missing outcomes.
- Members who died must remain in the analysis.
- Convert `date_of_death` into a binary indicator such as `death_within_90d_flag`, where a populated death date represents death within the applicable 90-day period. Drop the raw death date after creating the flag so that a raw date is never modeled.
- Do not exclude members merely because the death indicator equals 1.
- Preserve the original `member_id` in scored outputs, but never use it as a predictor.
- Do not exclude a row merely because the source `intervention_flag` is missing when `OptOut_flag` identifies it as control. Only contradictory or unresolved treatment rows may be excluded, after reporting their counts.
- Validate that both treatment groups and both binary outcome classes are present, including within the treated and control training subsets required for cross-validation.

## Post-treatment leakage exclusions

The following variables must never appear in any model matrix, feature list, encoded feature, SHAP result, variable-importance table, or predictor plot:

- `programStart`
- `programEnd`
- `engaged_date`
- `engagement_length`
- `InterventionCount`
- `Successful_Intervention`
- `SuccessfulInterventions`
- `outcome_ed_30d`
- `outcome_ed_6MO`
- `outcome_admit_30d`
- `outcome_admit_90d`
- `outcome_admit_6MO`
- `outcome_total_cost_90d`
- `outcome_total_cost_6MO`
- raw `OptOut_flag` as a model predictor (retain it for treatment derivation, audits, and scored outputs)

Column names in the actual CSV include spelling and case variations such as `ProgramStart`, `ProgramEnd`, `InterventionCount`, and `SuccessfulInterventions`. Apply exclusions after normalizing column names and make the checks robust to case, underscores, and common singular/plural variations.

Also exclude any other variable determined by, collected after, or directly revealing treatment assignment. Use baseline predictors only. Do not derive features from `ProgramStart`, `ProgramEnd`, engagement dates, or other post-treatment dates before dropping them.

## Additional exclusion

Drop `total_cost_6MO` if that column exists. Do not accidentally drop the baseline variable `total_cost_last_6m`; that is a different, pretreatment variable and may remain as a predictor.

## Preprocessing requirements

Adapt only the data-loading and preprocessing portions needed for the Funds schema. Preserve the source notebook's modeling methods.

Use an explicit baseline predictor allowlist or an equally strict leakage-safe selection process. Based on the current Funds schema, evaluate appropriate pretreatment variables such as:

- demographics and coverage
- `client_contract`
- age and gender
- `dual_elg`
- county and state
- plan type
- baseline clinical-condition flags
- baseline behavioral-health flags
- baseline social-needs flags
- baseline utilization measures
- baseline cost measures
- baseline medication measures
- baseline risk scores and risk tier
- the derived binary death flag

Do not model `member_id`. Handle categorical variables, binary flags, missing numeric values, missing categorical values, and zero-variance predictors consistently with the source notebook.

Produce a preprocessing audit that reports:

- raw row and column counts
- normalized source-column names
- missing treatment count
- missing outcome count before filling
- number of outcome values filled with zero
- treatment, raw count-outcome, and final binary-outcome distributions
- death-flag distribution
- predictors retained
- columns excluded and the reason for each exclusion
- duplicate-member count
- rows in the train and test sets
- treated/control and outcome counts in each split
- zero-variance columns removed
- final encoded feature count

Add executable assertions confirming that none of the prohibited post-treatment variables, or encoded derivatives of them, are present in the final model matrix.

## Modeling parity

Keep the same methods and settings as the source notebook unless a change is strictly necessary because of the Funds schema. This includes:

- 70/30 train/test split
- random seed 123
- stratification by treatment and binary outcome
- separate treated and control outcome models
- XGBoost T-learner
- the identical XGBoost grid:
  - `max_depth`: 3, 4, 5
  - `eta`: 0.03, 0.05, 0.10
  - `min_child_weight`: 1, 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
- five-fold stratified cross-validation when supported by class counts
- maximum 500 boosting rounds
- early stopping after 20 rounds
- selection by validation AUC
- GLMNet-style elastic-net logistic comparison using the same alpha grid, lambda/C search, scaling, folds, seed, and evaluation rules
- T-learner scoring definitions and benefit-score direction
- X-learner methods and propensity modeling
- all factual model diagnostics
- factual model AUC and event-class diagnostics
- Brier scores for the treated and control factual predictions
- probability-calibration tables by predicted-risk decile
- calibration summaries and predicted-versus-observed calibration charts
- decile summaries
- observed treated/control outcome gaps
- uplift curves
- variable importance
- SHAP analyses
- risk-tier versus benefit-group analyses
- cumulative and marginal targeting analyses
- ROI calculations and charts
- full-file scoring
- the same chart styles, titles, dimensions, and file formats, updated only where necessary to identify the Funds analysis

Do not simplify, remove, or silently substitute modeling methods. Preserve the source notebook's GPU requirement and XGBoost configuration. If CUDA is unavailable, do not silently switch algorithms or parameters; implement the notebook correctly and clearly report the execution limitation.

## Dataset-specific exceptions

The source notebook contains synthetic true-benefit validation that relies on known synthetic treatment-effect ground truth. Do not fabricate equivalent ground truth for Funds Combined.

Retain a clearly labeled section in the duplicated notebook explaining that synthetic true-benefit validation is not applicable because this dataset does not provide known counterfactual treatment effects. Skip only calculations that genuinely require unavailable synthetic ground truth. All other modeling, validation, reporting, charts, and outputs must still run.

## Output structure and isolation

Create a new output root named exactly:

`Outputs/Uplift_Funds/`

It must mirror the existing PRP output structure under `Outputs/Uplift/`. The Funds notebook must write its artifacts to equivalent relative paths under the new root.

At minimum, create this structure:

```text
Outputs/Uplift_Funds/
|-- Python/
|   |-- Predictor_Distributions/
|   |   |-- Categorical/
|   |   `-- Numeric/
|   |-- T-Learner/
|   |   |-- GLMNet/
|   |   `-- XGBoost/
|   `-- X-Learner/
|       |-- GLMNet/
|       `-- XGBoost/
`-- R/
```

The Python notebook's output root must therefore be:

`Outputs/Uplift_Funds/Python/`

Use the same artifact filenames and relative locations as the corresponding PRP outputs wherever the analysis is applicable. For example:

- PRP: `Outputs/Uplift/Python/T-Learner/XGBoost/uplift_scored_output.csv`
- Funds: `Outputs/Uplift_Funds/Python/T-Learner/XGBoost/uplift_scored_output.csv`

This mirroring requirement applies to:

- root-level data-review and model-evaluation summaries
- predictor data dictionaries and summary tables
- individual categorical and numeric predictor charts
- combined predictor-distribution PDFs
- T-Learner XGBoost artifacts
- T-Learner GLMNet artifacts
- X-Learner XGBoost artifacts
- X-Learner GLMNet artifacts
- calibration outputs
- Brier-score outputs
- decile summaries and charts
- uplift curves
- observed-gap analyses
- ROI analyses
- targeting and savings analyses
- risk-tier analyses
- variable-importance outputs
- SHAP outputs
- full scored-member outputs
- model comparison and recommendation summaries

Do not copy PRP result files into `Uplift_Funds`. Every Funds artifact must be freshly generated from `Funds_combined.csv`.

Do not reproduce incidental files or directories such as `.ipynb_checkpoints`, Python caches, or stale `.gitkeep` files.

The `R/` directory should be created to preserve the PRP root structure, but do not fabricate R results if this task only duplicates the Python notebook.

Do not write any Funds result into `Outputs/Uplift/`, and do not overwrite, rename, or delete any existing PRP output.

Do not run a repository-wide README regeneration script if it would overwrite PRP documentation. If documentation is needed, create a Funds-specific output index or README instead.

## Output parity manifest

The PRP output structure contains files related to synthetic true-benefit validation. Because Funds Combined does not contain known counterfactual ground truth, do not create misleading versions of those files.

Instead, create:

`Outputs/Uplift_Funds/Python/output_parity_manifest.csv`

This manifest must contain:

- corresponding PRP relative path
- expected Funds relative path
- artifact category
- applicability status
- generated status
- explanation when not applicable

Synthetic-ground-truth artifacts should be marked `not_applicable` with an explanation. All other applicable PRP artifacts must have corresponding Funds artifacts.

## Execution and verification

After creating the notebook:

1. Execute it from top to bottom using the repository's configured environment.
2. Fix implementation or runtime errors that are within scope.
3. Ensure the saved notebook contains fresh outputs from the Funds dataset.
4. Verify every expected applicable CSV, PNG, and PDF artifact exists.
5. Confirm that no existing PRP output was overwritten.
6. Search the final notebook, feature list, model matrices, scored files, importance tables, and SHAP outputs for prohibited variables and report the results.
7. Confirm that missing `outcome_ed_90d` values became zero.
8. Confirm that deceased members remain in the modeling population and the raw death date is absent from predictors.
9. Compare the source and duplicate notebook section by section and document any intentional differences.
10. Recursively compare `Outputs/Uplift/Python/` against `Outputs/Uplift_Funds/Python/` and verify that every applicable PRP artifact category and relative directory has a freshly generated Funds counterpart.
11. Review `git diff` and ensure only Funds-specific files were changed or added.

## Completion report

Deliver a concise completion summary containing:

- the new notebook path
- the input path actually used
- the `Outputs/Uplift_Funds/` output root
- the complete Funds output directory tree
- row counts and treatment/outcome distributions
- the final predictor list
- confirmation that leakage checks passed
- confirmation that the notebook executed successfully
- the number of applicable artifacts expected
- the number successfully generated
- any missing artifacts
- every artifact marked not applicable and why
- confirmation that no file under `Outputs/Uplift/` was modified
- all files changed or created
