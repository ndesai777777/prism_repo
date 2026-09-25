# Oregon Uplift Model Implementation Prompt

Work in the current `prism_repo` repository and implement an Oregon version of the existing Funds Combined uplift-model analysis.

## Primary goal

Use the existing notebook:

`Code/Uplift Model Code_Funds_Combined.ipynb`

as the authoritative template. Create a duplicate notebook that performs the equivalent analysis using:

`DataSets/Oregon_2YearDataset.csv`

The Oregon notebook must reproduce the same analysis as the Funds Combined notebook, including:

- Exploratory analysis
- Tables and descriptive statistics
- Graphs and charts
- Train/test methodology
- Uplift-modeling approach
- Model specifications
- Hyperparameter tuning
- Cross-validation
- Evaluation metrics
- Feature-importance analysis
- Uplift, Qini, and gain curves
- Random seeds
- Final conclusions

Keep the same cell order, narrative structure, modeling choices, and visualization style wherever the Oregon data supports them. Do not modify or overwrite the original Funds Combined notebook.

## Deliverables

1. Create:

   `Code/Uplift Model Code_Oregon_2YearDataset.ipynb`

2. Store Oregon-specific generated outputs separately from the Funds outputs, preferably under:

   `Outputs/Uplift_Oregon/Python/`

3. After the notebook executes successfully, create an Oregon-specific analysis report modeled on the existing **PRISM Intervention Benefit Funds** README. Give the Oregon report a distinct filename and do not overwrite the Funds README.

## Data inspection and preprocessing

Before changing the copied notebook:

1. Inspect the complete Oregon dataset schema, data types, missingness, unique values, and row count.
2. Inspect the original Funds notebook cell by cell to understand every transformation and modeling decision.
3. Build an explicit mapping between the variables expected by the Funds notebook and the available Oregon variables.
4. Adapt only the preprocessing needed for differences in the Oregon schema. Preserve the analytical and modeling methodology.
5. Do not invent a column mapping based only on similar names. Verify mappings using value distributions, data types, and the original notebook's logic.
6. Add a clearly labeled notebook section documenting:
   - Original Oregon column name
   - Standardized or modeling name
   - Role: treatment, outcome, eligible predictor, excluded identifier, or post-treatment variable
   - Transformation applied
   - Reason for inclusion or exclusion

## Treatment assignment

The Oregon data contains treatment-related variables such as:

- `intervention_flag`
- `OptOut_flag` or `optoutflag`
- `engaged_flag`

Requirements:

- Use the same treatment definition used conceptually in the Funds Combined analysis.
- Treat `intervention_flag` as the canonical treatment indicator if it is present and valid.
- `OptOut_flag` contains redundant treatment information and must not be included as a model predictor.
- Drop `OptOut_flag` or `optoutflag` after using it only for validation, if needed.
- Inspect and document the relationship among `intervention_flag`, `OptOut_flag`, and `engaged_flag`.
- Verify which value represents treatment and which represents control using the data and the original notebook. Do not silently guess or reverse treatment coding.
- Convert the final treatment variable to an explicit binary 0/1 indicator.
- Report treatment and control counts and any disagreements among the treatment-related fields.
- Do not use `engaged_flag` as a predictor if it reflects post-assignment engagement rather than treatment assignment.
- Never include the final treatment variable or a treatment proxy in the predictor feature matrix.

## Outcome

The outcome of interest is:

`outcome_ed_90d`

Requirements:

- Convert it into a binary outcome.
- If it is already binary, preserve valid 0/1 values.
- If it is a count, convert values greater than zero to 1 and zero to 0.
- Treat missing or NA values as 0, as requested.
- Validate and report the outcome distribution before and after conversion.
- Do not use `outcome_ed_90d` as a model feature.
- Exclude direct derivatives, alternate encodings, or proxies for the outcome from the feature matrix.

## Post-treatment leakage prevention

The following variables are post-treatment variables and must be removed before imputation, encoding, feature selection, hyperparameter tuning, training, or evaluation:

- `programStart`
- `programEnd`
- `engaged_date`
- `engagement_length`
- `InterventionCount`
- `INterventionCount`
- `Successful_Intervention`
- `outcome_ed_30d`
- `outcome_ed_6MO`
- `outcome_admit_30d`
- `outcome_admit_90d`
- `outcome_admit_6MO`
- `outcome_total_cost_90d`
- `outcome_total_cost_6MO`

Resolve these names case-insensitively so capitalization differences do not allow a prohibited variable into the model.

Also inspect the dataset for aliases or other columns that contain equivalent post-intervention information. Exclude variables derived from treatment participation, engagement, interventions delivered, or outcomes observed after treatment, even if they are not named exactly as listed above.

Add an automated leakage assertion immediately before model fitting. The notebook must fail with a clear message if any prohibited post-treatment variable, treatment proxy, target variable, or direct derivative remains in the feature matrix.

## Death handling

- Do not exclude a person merely because they died within 90 days of treatment.
- Exclude all death-date variables from the model and preprocessing pipeline.
- Do not derive a death indicator or any other predictor from death information.
- Confirm that retaining deceased individuals does not otherwise change their outcome coding.
- Missing `outcome_ed_90d` values must still be coded as 0.
- Document the number of retained records with available death information for validation purposes only, without using death information in model training, feature selection, tuning, or evaluation stratification.

## General cleaning requirements

- Preserve one row per appropriate analytical unit and investigate duplicates before removing any.
- Document exclusions and row-count changes at every cleaning step.
- Exclude IDs, names, free-text fields, raw dates, and other non-generalizable identifiers from model predictors unless the original methodology explicitly requires them.
- Handle missing values using the same approach as the Funds notebook where appropriate.
- Fit imputers, encoders, scalers, feature selectors, and other preprocessing only on training data to prevent leakage.
- Use pipelines where practical.
- Check categorical levels, invalid values, impossible dates, constant columns, near-empty columns, and numerical outliers.
- Do not drop rows simply because a predictor is missing unless that matches the original methodology and is justified.
- Preserve the original random seeds and reproducibility settings.
- Do not modify the source Oregon CSV.

## Modeling fidelity

The original notebook is the source of truth for the analytical method. Preserve:

- Treatment and outcome modeling framework
- Algorithms
- Hyperparameter search spaces
- Search method
- Cross-validation strategy
- Scoring criteria
- Train/test split
- Random seeds
- Thresholds
- Evaluation metrics
- Plot types and formatting
- Model-comparison logic

Only make a modeling change if the Oregon schema makes the original method impossible. If that happens, document:

1. The original method.
2. Why it cannot be applied.
3. The smallest necessary adaptation.
4. The effect on comparability.

Do not quietly skip a model, chart, metric, or notebook section when data is insufficient. Display a clear explanation in the corresponding section.

## Report requirements

After the Oregon notebook has executed successfully, create an Oregon-specific Markdown report following the organization, level of detail, tables, visuals, and narrative style of the existing **PRISM Intervention Benefit Funds** README.

The report must:

- Use Oregon results only.
- Clearly describe the treatment and outcome definitions.
- Explain the preprocessing and leakage controls.
- Summarize the analytical population.
- Present the same categories of model results and visualizations as the Funds report where supported.
- Clearly label any analysis that could not be reproduced and explain why.
- Use relative repository paths for embedded Oregon output images.
- Avoid making causal claims beyond what the design and modeling methodology support.
- Not overwrite or alter the Funds report.

## Validation

Execute the completed Oregon notebook from top to bottom in a clean kernel.

Confirm that:

- Every code cell runs without error.
- All expected charts and tables are rendered.
- The original Funds notebook is unchanged.
- The Oregon dataset is unchanged.
- No post-treatment variable appears in any model feature matrix.
- No death date or death-derived variable appears in the model feature matrix.
- Treatment and outcome coding are explicitly validated.
- Missing `outcome_ed_90d` values become 0.
- People who died within 90 days remain in the analytical population.
- Preprocessing is fitted without train/test leakage.
- Outputs are written only to the Oregon-specific output directory.
- Results are reproducible from a clean execution.

Add explicit assertions or validation cells for these requirements instead of relying only on visual inspection.

## Completion summary

Before finishing, provide a concise implementation summary containing:

- Files created or modified
- Final analytical sample size
- Treatment and control counts
- Positive and negative outcome counts
- Exact treatment definition used
- Number of model features
- Full list of excluded post-treatment fields found
- Any requested fields not found
- Death-related fields excluded
- Any assumptions or deviations from the Funds notebook
- Notebook execution status
- Tests and validation checks performed

## Working constraints

- Do not overwrite or edit the original Funds notebook.
- Do not commit Jupyter `.ipynb_checkpoints` files.
- Preserve unrelated existing repository changes.
- Do not commit or push changes unless explicitly asked.
- Favor transparent, reviewable preprocessing over silent assumptions.
- Do not stop after merely creating a notebook copy. Complete, execute, validate, and document the Oregon analysis.
