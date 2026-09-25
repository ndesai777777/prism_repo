"""Build the Oregon 2-Year uplift notebook from the executed Funds template.

The Funds notebook is the modeling specification. This builder changes only the
dataset-specific loading, cleaning, treatment derivation, leakage controls,
train/test grouping, train-fitted preprocessing, output paths, and Oregon labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Code" / "Uplift Model Code_Funds_Combined.ipynb"
TARGET = ROOT / "Code" / "Uplift Model Code_Oregon_2YearDataset.ipynb"


def lines(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def set_cell(notebook: dict, index: int, text: str) -> None:
    notebook["cells"][index]["source"] = lines(text)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

# Start from the exact Funds analytical notebook, clear its execution state, and
# relabel Oregon-specific narrative/output references without touching the Funds file.
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    source = "".join(cell.get("source", []))
    source = source.replace("Uplift_Funds", "Uplift_Oregon")
    source = source.replace("Funds_combined.csv", "Oregon_2YearDataset.csv")
    source = source.replace("FUNDS COMBINED", "OREGON 2-YEAR")
    source = source.replace("Funds Combined", "Oregon 2-Year")
    source = source.replace("funds_specific", "oregon_specific")
    source = source.replace("Funds-specific", "Oregon-specific")
    source = source.replace("Funds", "Oregon")
    source = source.replace("funds_csv_path", "oregon_csv_path")
    source = source.replace("source_intervention_flag", "source_engaged_flag")
    # NumPy 2.4 removed the old alias; trapezoid is the exact replacement.
    source = source.replace("np.trapz", "np.trapezoid")
    cell["source"] = source.splitlines(keepends=True)

set_cell(
    notebook,
    0,
    """
    # OREGON 2-YEAR UPLIFT MODEL (T-LEARNER WITH XGBOOST)

    Input: `DataSets/Oregon_2YearDataset.csv`  
    Binary outcome: `outcome_ed_90d` (1 = one or more ED visits; missing = 0)  
    Treatment: `Engaged_flag` (1 = treated/engaged, 0 = control/opted out), validated against `OptOut_flag`

    This notebook preserves the Funds Combined notebook's modeling, hyperparameter
    search, diagnostics, calibration, Brier scoring, charts, and output methods.
    Oregon-specific changes are limited to schema-aware cleaning, treatment derivation,
    post-treatment leakage controls, member-grouped splitting, train-fitted preprocessing,
    and isolated Oregon outputs.

    XGBoost is configured to require the SageMaker CUDA GPU on device 0, matching
    the authoritative Funds notebook. Run this notebook on a GPU-backed instance.

    The Funds source notebook does not define separate Qini or cumulative-gain
    calculations. Its cumulative uplift-by-targeted-fraction analysis is reproduced
    as written; no new metric was invented because that would change the source method.

    Repeated member episodes are retained as analytical records. Every member is kept
    entirely in either training or testing to prevent identity leakage. Death fields are
    not available in this source file; no row is excluded based on death, and no
    death-derived predictor is created.
    """,
)

cell4 = "".join(notebook["cells"][4]["source"])
cell4 = cell4.replace(
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold\n",
)
cell4 = cell4.replace(
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import StandardScaler\nfrom IPython.display import display\n",
)
cell4 = cell4.replace(
    "    clean_names_simple,\n",
    "    clean_feature_names,\n    clean_names_simple,\n",
)
set_cell(notebook, 4, cell4)

set_cell(
    notebook,
    6,
    """
    oregon_csv_path = PROJECT_ROOT / 'DataSets' / 'Oregon_2YearDataset.csv'
    if not oregon_csv_path.exists():
        nested_candidate = PROJECT_ROOT / 'prism_repo' / 'DataSets' / 'Oregon_2YearDataset.csv'
        if nested_candidate.exists():
            oregon_csv_path = nested_candidate
        else:
            raise FileNotFoundError(
                'Could not locate Oregon_2YearDataset.csv in either expected repository location.'
            )

    output_root = ensure_output_folder(PROJECT_ROOT / 'Outputs' / 'Uplift_Oregon')
    output_folder = ensure_output_folder(output_root / 'Python')
    r_output_folder = ensure_output_folder(output_root / 'R')
    tlearner_root_folder = ensure_output_folder(output_folder / 'T-Learner')
    xgboost_output_folder = ensure_output_folder(tlearner_root_folder / 'XGBoost')
    glmnet_output_folder = ensure_output_folder(tlearner_root_folder / 'GLMNet')

    xgboost_output_path = xgboost_output_folder / 'uplift_scored_output.csv'
    xgboost_summary_path = xgboost_output_folder / 'uplift_decile_summary.csv'
    glmnet_output_path = glmnet_output_folder / 'uplift_scored_output.csv'
    glmnet_summary_path = glmnet_output_folder / 'uplift_decile_summary.csv'
    output_path = xgboost_output_path
    summary_path = xgboost_summary_path

    print('Oregon data path:', oregon_csv_path)
    print('Project root resolved to:', PROJECT_ROOT)
    print('Oregon output root:', output_root)
    print('XGBoost outputs will be saved to:', xgboost_output_folder)
    print('GLMNET outputs will be saved to:', glmnet_output_folder)
    """,
)

set_cell(
    notebook,
    10,
    """
    df_raw = pd.read_csv(oregon_csv_path, low_memory=False)
    original_column_names = list(df_raw.columns)
    df_raw.columns = clean_names_simple(df_raw.columns)
    normalized_column_names = list(df_raw.columns)
    if pd.Index(df_raw.columns).duplicated().any():
        duplicates = pd.Index(df_raw.columns)[pd.Index(df_raw.columns).duplicated()].tolist()
        raise ValueError(f'Duplicate columns after normalization: {duplicates}')

    original_to_normalized = pd.DataFrame({
        'original_oregon_column': original_column_names,
        'normalized_column': normalized_column_names,
    })

    print('Rows:', len(df_raw))
    print('Columns:', len(df_raw.columns))
    print('Original-to-normalized columns:')
    display(original_to_normalized)
    """,
)

set_cell(
    notebook,
    12,
    """
    required_fields = ['member_id', 'outcome_ed_90d', 'engaged_flag', 'optout_flag']
    require_columns(df_raw, required_fields)
    """,
)

set_cell(
    notebook,
    14,
    """
    df = df_raw.copy()
    raw_row_count = len(df)
    unique_member_count = int(df['member_id'].nunique(dropna=True))
    duplicate_member_count = int(df['member_id'].duplicated().sum())
    exact_duplicate_row_count = int(df.duplicated().sum())

    source_engaged = to_binary(df['engaged_flag'])
    source_optout = to_binary(df['optout_flag'])
    source_engaged = source_engaged.where(source_engaged.isin([0.0, 1.0]))
    source_optout = source_optout.where(source_optout.isin([0.0, 1.0]))

    resolved_treated = source_engaged.eq(1.0) & source_optout.eq(0.0)
    resolved_control = source_engaged.eq(0.0) & source_optout.eq(1.0)
    both_present = source_engaged.notna() & source_optout.notna()
    contradictory_treatment = both_present & ~(resolved_treated | resolved_control)

    df['source_engaged_flag'] = source_engaged
    df['optout_flag'] = source_optout
    df['intervention_flag'] = np.select(
        [resolved_treated, resolved_control], [1.0, 0.0], default=np.nan
    )
    df['treatment_derivation_status'] = np.select(
        [contradictory_treatment, resolved_treated, resolved_control],
        ['contradictory', 'treated', 'control'],
        default='unresolved',
    )

    outcome_numeric = pd.to_numeric(df['outcome_ed_90d'], errors='coerce')
    missing_outcome_before_fill = int(outcome_numeric.isna().sum())
    if (outcome_numeric.dropna() < 0).any():
        raise ValueError('outcome_ed_90d contains negative values.')
    raw_outcome_count_distribution = outcome_numeric.fillna(0.0).value_counts().sort_index().to_dict()
    positive_counts_binarized = int((outcome_numeric.fillna(0.0) > 0).sum())
    df['outcome_ed_90d'] = (outcome_numeric.fillna(0.0) > 0).astype(float)

    # ProgramFinal is the Oregon schema counterpart to the Funds program field.
    # It is copied only into the standardized analysis frame; the source CSV is unchanged.
    if 'programfinal' in df.columns and 'program' not in df.columns:
        df['program'] = df['programfinal']

    risk_numeric = pd.to_numeric(df['risk_tier'], errors='coerce')
    missing_risk_tier_rows = int(risk_numeric.isna().sum())
    invalid_risk_tier_rows = int((risk_numeric.notna() & ~risk_numeric.isin([1, 2, 3, 4, 5])).sum())
    valid_risk_tier = risk_numeric.where(risk_numeric.isin([1, 2, 3, 4, 5]))
    df['risk_tier'] = valid_risk_tier.astype('Int64').astype('string').fillna('Missing')

    death_related_fields_found = [
        column for column in df_raw.columns
        if 'death' in canonical_column_name(column) or 'deceased' in canonical_column_name(column)
    ] if 'canonical_column_name' in globals() else [
        column for column in df_raw.columns
        if 'death' in str(column).lower() or 'deceased' in str(column).lower()
    ]
    rows_with_available_death_information = 0
    if death_related_fields_found:
        death_available_mask = pd.Series(False, index=df_raw.index)
        for column in death_related_fields_found:
            values = df_raw[column].astype('string').str.strip()
            death_available_mask |= values.notna() & values.ne('')
        rows_with_available_death_information = int(death_available_mask.sum())

    print('Engaged_flag versus OptOut_flag validation:')
    display(pd.crosstab(
        df['source_engaged_flag'].fillna('Missing'),
        df['optout_flag'].fillna('Missing'),
        dropna=False,
    ))
    print('Treatment derivation status:')
    display(df['treatment_derivation_status'].value_counts(dropna=False).rename_axis('status').to_frame('n'))
    print('Missing outcomes filled with zero:', missing_outcome_before_fill)
    print('Records with one or more ED visits after binarization:', positive_counts_binarized)
    print('Unique members:', unique_member_count)
    print('Repeated member episodes retained:', duplicate_member_count)
    print('Exact duplicate rows:', exact_duplicate_row_count)
    print('Oregon risk-tier distribution (1 = lowest, 5 = highest):')
    display(df['risk_tier'].value_counts(dropna=False).reindex(['1', '2', '3', '4', '5', 'Missing'], fill_value=0))
    print('Missing source risk tiers:', missing_risk_tier_rows)
    print('Out-of-range source risk tiers treated as unassigned:', invalid_risk_tier_rows)
    print('Death-related source fields found:', death_related_fields_found)
    print('Rows with available death information (reporting only):', rows_with_available_death_information)
    """,
)

set_cell(
    notebook,
    16,
    """
    # No features are derived from program, engagement, intervention, outcome, or death dates.
    # These are post-treatment or otherwise prohibited fields and are excluded before modeling.
    print('Post-treatment and death-related feature engineering intentionally skipped for Oregon 2-Year.')
    """,
)

set_cell(
    notebook,
    18,
    """
    import re


    def canonical_column_name(value):
        return re.sub(r'[^a-z0-9]+', '', str(value).lower())


    candidate_predictors_all = [
        'client_contract', 'age', 'gender', 'dual_elg', 'county', 'state', 'plan_type',
        'diabetes_flag', 'chf_flag', 'copd_flag', 'asthma_flag', 'depression_flag',
        'anxiety_flag', 'substance_use_flag', 'ckd_flag', 'bh_flag', 'htn_flag',
        'hyperlipidemia_flag', 'cad_flag', 'financial_strain_flag', 'food_insecurity_flag',
        'housing_instability_flag', 'transportation_barrier_flag',
        'healthliteracy_challenges_flag', 'op_visits_last_6m', 'ed_visits_last_30d',
        'ed_visits_last_6m', 'admits_last_6m', 'observation_stays_last_6m',
        'total_cost_last_6m', 'rx_count_last_6m', 'opioid_flag', 'polypharmacy_flag',
        'current_risk_score', 'percolator_score', 'risk_tier', 'program',
    ]
    prohibited_names = [
        'programstart', 'programend', 'engaged_date', 'engagement_length',
        'interventioncount', 'successful_intervention', 'successfulinterventions',
        'outcome_ed_30d', 'outcome_ed_6mo', 'outcome_admit_30d', 'outcome_admit_90d',
        'outcome_admit_6mo', 'outcome_total_cost_90d', 'outcome_total_cost_6mo',
        'total_cost_6mo',
    ]
    prohibited_canonical = {canonical_column_name(column) for column in prohibited_names}
    raw_treatment_canonical = {
        canonical_column_name('source_engaged_flag'),
        canonical_column_name('engaged_flag'),
        canonical_column_name('optout_flag'),
    }
    target_canonical = {canonical_column_name('outcome_ed_90d')}
    death_canonical = {
        canonical_column_name(column) for column in df.columns
        if 'death' in canonical_column_name(column) or 'deceased' in canonical_column_name(column)
    }

    candidate_predictors = [column for column in candidate_predictors_all if column in df.columns]
    missing_predictors = [column for column in candidate_predictors_all if column not in df.columns]
    exclusion_reasons = {}
    for column in df.columns:
        canonical = canonical_column_name(column)
        if column in candidate_predictors:
            continue
        if (
            canonical in prohibited_canonical
            or (column.startswith('outcome_') and column != 'outcome_ed_90d')
            or canonical in death_canonical
        ):
            exclusion_reasons[column] = 'post-treatment, death-related, alternate outcome, or explicitly prohibited'
        elif canonical in raw_treatment_canonical or column == 'intervention_flag':
            exclusion_reasons[column] = 'treatment definition/source; retained as metadata but excluded from predictors'
        elif column == 'member_id':
            exclusion_reasons[column] = 'identifier retained in outputs but excluded from predictors'
        elif column == 'treatment_derivation_status':
            exclusion_reasons[column] = 'treatment derivation audit metadata'
        elif column == 'programfinal':
            exclusion_reasons[column] = 'source column standardized to program; source duplicate excluded'
        else:
            exclusion_reasons[column] = 'not on leakage-safe baseline predictor allowlist'

    meta_columns = [
        'member_id', 'source_engaged_flag', 'optout_flag',
        'treatment_derivation_status', 'outcome_ed_90d', 'intervention_flag',
    ]
    model_df = df[[*meta_columns, *candidate_predictors]].copy()
    unresolved_mask = ~model_df['treatment_derivation_status'].isin(['treated', 'control'])
    excluded_treatment_rows = int(unresolved_mask.sum())
    model_df = model_df.loc[~unresolved_mask].copy().reset_index(drop=True)

    found_post_treatment_fields = [
        column for column in df_raw.columns
        if canonical_column_name(column) in prohibited_canonical
        or (column.startswith('outcome_') and column != 'outcome_ed_90d')
    ]

    if missing_predictors:
        print('Allowlisted predictors not found:', missing_predictors)
    print('Rows excluded because treatment could not be resolved:', excluded_treatment_rows)
    print('Post-treatment fields found and excluded:', found_post_treatment_fields)
    print('Candidate baseline predictors before train-only filtering:', candidate_predictors)
    """,
)

set_cell(
    notebook,
    21,
    """
    binary_predictors = [
        'dual_elg', 'diabetes_flag', 'chf_flag', 'copd_flag', 'asthma_flag',
        'depression_flag', 'anxiety_flag', 'substance_use_flag', 'ckd_flag', 'bh_flag',
        'htn_flag', 'hyperlipidemia_flag', 'cad_flag', 'financial_strain_flag',
        'food_insecurity_flag', 'housing_instability_flag', 'transportation_barrier_flag',
        'healthliteracy_challenges_flag', 'opioid_flag', 'polypharmacy_flag',
    ]
    for column in present_columns(binary_predictors, model_df):
        model_df[column] = to_binary(model_df[column])

    possible_numeric_cols = [
        'age', 'op_visits_last_6m', 'ed_visits_last_30d', 'ed_visits_last_6m',
        'admits_last_6m', 'observation_stays_last_6m', 'total_cost_last_6m',
        'rx_count_last_6m', 'current_risk_score', 'percolator_score',
    ]

    def parse_currency(values):
        cleaned = (
            values.astype('string')
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors='coerce')


    for column in present_columns(possible_numeric_cols, model_df):
        if column == 'total_cost_last_6m':
            model_df[column] = parse_currency(model_df[column])
        else:
            model_df[column] = pd.to_numeric(model_df[column], errors='coerce')

    # Missing values remain missing here. Imputation values, category levels, and
    # zero-variance filtering are fitted only after the grouped train/test split.
    model_df = model_df[[
        'member_id', 'source_engaged_flag', 'optout_flag', 'treatment_derivation_status',
        'outcome_ed_90d', 'intervention_flag', *candidate_predictors,
    ]].copy()

    print('Pre-split modeling columns (missing values intentionally retained):')
    print(list(model_df.columns))
    """,
)

set_cell(
    notebook,
    23,
    """
    def grouped_stratified_train_test_split(
        frame,
        group_column='member_id',
        train_fraction=0.70,
        seed=123,
        n_candidates=500,
    ):
        strata = frame[['intervention_flag', 'outcome_ed_90d']].astype(int).astype(str).agg('_'.join, axis=1)
        overall_distribution = strata.value_counts(normalize=True).sort_index()
        splitter = GroupShuffleSplit(
            n_splits=n_candidates,
            train_size=train_fraction,
            random_state=seed,
        )
        best = None
        best_score = np.inf
        all_strata = set(overall_distribution.index)

        for train_positions, test_positions in splitter.split(frame, strata, groups=frame[group_column]):
            train_strata = strata.iloc[train_positions]
            test_strata = strata.iloc[test_positions]
            if set(train_strata.unique()) != all_strata or set(test_strata.unique()) != all_strata:
                continue
            train_distribution = train_strata.value_counts(normalize=True).reindex(overall_distribution.index, fill_value=0)
            test_distribution = test_strata.value_counts(normalize=True).reindex(overall_distribution.index, fill_value=0)
            size_error = abs((len(train_positions) / len(frame)) - train_fraction)
            balance_error = (
                (train_distribution - overall_distribution).abs().sum()
                + (test_distribution - overall_distribution).abs().sum()
            )
            score = float(size_error + balance_error)
            if score < best_score:
                best_score = score
                best = (train_positions, test_positions)

        if best is None:
            raise ValueError('Could not create a member-grouped split containing all treatment/outcome strata.')

        train_positions, test_positions = best
        train = frame.iloc[train_positions].copy().reset_index(drop=True)
        test = frame.iloc[test_positions].copy().reset_index(drop=True)
        overlap = set(train[group_column]).intersection(test[group_column])
        if overlap:
            raise AssertionError(f'Member leakage across train/test: {sorted(overlap)[:10]}')
        return train, test, best_score


    train_df, test_df, split_balance_score = grouped_stratified_train_test_split(
        model_df,
        group_column='member_id',
        train_fraction=0.70,
        seed=123,
    )

    train_test_member_overlap = set(train_df['member_id']).intersection(test_df['member_id'])
    assert not train_test_member_overlap

    print('Training records:', len(train_df))
    print('Testing records:', len(test_df))
    print('Training unique members:', train_df['member_id'].nunique())
    print('Testing unique members:', test_df['member_id'].nunique())
    print('Train/test member overlap:', len(train_test_member_overlap))
    print('Grouped split balance score:', round(split_balance_score, 6))
    display(
        model_df.groupby(['intervention_flag', 'outcome_ed_90d'], dropna=False)
        .size().rename('n').reset_index()
    )
    """,
)

set_cell(
    notebook,
    27,
    """
    import json

    id_cols = ['member_id', 'source_engaged_flag', 'optout_flag', 'treatment_derivation_status']

    # Feature selection, imputation statistics, and category levels are learned from
    # training records only. This prevents information from the held-out set from
    # influencing preprocessing.
    train_unique_counts = train_df[candidate_predictors].nunique(dropna=True)
    zero_variance_predictors = train_unique_counts[train_unique_counts <= 1].index.tolist()
    feature_cols = [column for column in candidate_predictors if column not in zero_variance_predictors]
    for column in zero_variance_predictors:
        exclusion_reasons[column] = 'zero-variance or all-missing predictor removed using training data only'

    numeric_feature_cols = [
        column for column in feature_cols
        if column in possible_numeric_cols or column in binary_predictors
    ]
    categorical_feature_cols = [column for column in feature_cols if column not in numeric_feature_cols]

    train_numeric_medians = {}
    for column in numeric_feature_cols:
        values = pd.to_numeric(train_df[column], errors='coerce')
        median = values.median(skipna=True)
        train_numeric_medians[column] = 0.0 if pd.isna(median) else float(median)

    train_category_levels = {}
    for column in categorical_feature_cols:
        values = train_df[column].astype('string')
        values = values.where(values.notna() & values.str.strip().ne(''), 'Missing')
        levels = sorted(values.astype(str).unique().tolist())
        if '__Unseen__' not in levels:
            levels.append('__Unseen__')
        train_category_levels[column] = levels


    def prepare_predictor_frame(frame):
        prepared = pd.DataFrame(index=frame.index)
        for column in numeric_feature_cols:
            values = pd.to_numeric(frame[column], errors='coerce')
            prepared[column] = values.fillna(train_numeric_medians[column]).astype(float)
        for column in categorical_feature_cols:
            values = frame[column].astype('string')
            values = values.where(values.notna() & values.str.strip().ne(''), 'Missing').astype(str)
            known = set(train_category_levels[column])
            values = values.where(values.isin(known), '__Unseen__')
            prepared[column] = pd.Categorical(values, categories=train_category_levels[column])
        return prepared[feature_cols]


    def design_matrix_from_prepared(prepared, expected_columns=None):
        matrix = pd.get_dummies(prepared, drop_first=False, dtype=float)
        matrix.columns = clean_feature_names(matrix.columns)
        matrix = matrix.astype(float)
        if expected_columns is not None:
            matrix = align_to_columns(matrix, expected_columns)
        return matrix


    train_prepared_df = prepare_predictor_frame(train_df)
    test_prepared_df = prepare_predictor_frame(test_df)
    train_matrix = design_matrix_from_prepared(train_prepared_df)
    test_matrix = design_matrix_from_prepared(test_prepared_df, train_matrix.columns)

    train_treated_mask = train_df['intervention_flag'].eq(1).to_numpy()
    train_control_mask = train_df['intervention_flag'].eq(0).to_numpy()
    x_treated = train_matrix.iloc[train_treated_mask].reset_index(drop=True)
    x_control = train_matrix.iloc[train_control_mask].reset_index(drop=True)
    x_test = test_matrix.reset_index(drop=True)
    combined_matrix = train_matrix.copy()
    y_treated = train_df.loc[train_treated_mask, 'outcome_ed_90d'].astype(float).to_numpy()
    y_control = train_df.loc[train_control_mask, 'outcome_ed_90d'].astype(float).to_numpy()

    forbidden_feature_canonical = (
        prohibited_canonical
        | raw_treatment_canonical
        | target_canonical
        | death_canonical
        | {canonical_column_name('intervention_flag')}
    )

    def is_forbidden_feature(column):
        canonical = canonical_column_name(column)
        if 'death' in canonical or 'deceased' in canonical:
            return True
        return any(canonical == item or canonical.startswith(item) for item in forbidden_feature_canonical)


    leaking_features = [column for column in feature_cols if is_forbidden_feature(column)]
    leaking_matrix_columns = [column for column in combined_matrix.columns if is_forbidden_feature(column)]
    assert not leaking_features, f'Prohibited predictors found: {leaking_features}'
    assert not leaking_matrix_columns, f'Encoded leakage columns found: {leaking_matrix_columns}'
    assert set(pd.Series(y_treated).unique()).issubset({0.0, 1.0})
    assert set(pd.Series(y_control).unique()).issubset({0.0, 1.0})
    assert 'member_id' not in feature_cols
    assert not set(train_df['member_id']).intersection(test_df['member_id'])
    assert len(model_df) == raw_row_count - excluded_treatment_rows
    assert int((df['outcome_ed_90d'] == 1).sum()) == positive_counts_binarized

    split_distribution = pd.concat([
        train_df.assign(split='train'), test_df.assign(split='test')
    ]).groupby(['split', 'intervention_flag', 'outcome_ed_90d'], dropna=False).size().rename('n').reset_index()
    split_distribution.to_csv(output_folder / 'preprocessing_split_distribution.csv', index=False)

    transformation_lookup = {
        'member_id': 'Retained as output identifier and grouping key; excluded from predictors.',
        'engaged_flag': 'Converted to binary and used as the canonical treatment source only.',
        'optout_flag': 'Converted to binary and used only to validate inverse treatment coding.',
        'outcome_ed_90d': 'Converted to numeric; missing set to 0; values greater than 0 mapped to 1.',
        'programfinal': 'Standardized to program; removed because it has no training variation.',
        'risk_tier': 'Values 1-5 retained as categories; missing or out-of-range values mapped to Missing.',
        'total_cost_last_6m': 'Currency symbols and commas removed; parsed as numeric; train-median imputation.',
    }

    column_audit_rows = []
    for original_name, normalized_name in zip(original_column_names, normalized_column_names):
        modeling_name = 'program' if normalized_name == 'programfinal' else normalized_name
        canonical = canonical_column_name(normalized_name)
        if normalized_name == 'engaged_flag':
            role = 'treatment source'
            status = 'excluded'
            reason = 'canonical treatment source; never a predictor'
        elif normalized_name == 'optout_flag':
            role = 'treatment validation source'
            status = 'excluded'
            reason = 'redundant inverse treatment signal; validation only'
        elif normalized_name == 'outcome_ed_90d':
            role = 'outcome'
            status = 'excluded'
            reason = 'binary target; never a predictor'
        elif normalized_name == 'member_id':
            role = 'identifier/grouping key'
            status = 'excluded'
            reason = 'retained for grouped splitting and outputs; never a predictor'
        elif normalized_name == 'case_manager_name':
            role = 'excluded identifier/all-missing field'
            status = 'excluded'
            reason = 'non-generalizable identifier field and entirely missing in Oregon'
        elif (
            canonical in prohibited_canonical
            or (normalized_name.startswith('outcome_') and normalized_name != 'outcome_ed_90d')
        ):
            role = 'post-treatment variable'
            status = 'excluded'
            reason = 'post-treatment leakage control'
        elif 'death' in canonical or 'deceased' in canonical:
            role = 'death-related variable'
            status = 'excluded'
            reason = 'death information is not used or transformed into a predictor'
        elif modeling_name in feature_cols:
            role = 'eligible baseline predictor'
            status = 'included'
            reason = 'leakage-safe baseline predictor retained after train-only filtering'
        elif modeling_name in zero_variance_predictors:
            role = 'candidate baseline predictor'
            status = 'excluded'
            reason = exclusion_reasons.get(modeling_name, 'zero variance in training data')
        else:
            role = 'excluded field'
            status = 'excluded'
            reason = exclusion_reasons.get(normalized_name, 'not on leakage-safe baseline predictor allowlist')

        if normalized_name in transformation_lookup:
            transformation = transformation_lookup[normalized_name]
        elif modeling_name in numeric_feature_cols:
            transformation = 'Numeric coercion and train-median imputation.'
        elif modeling_name in categorical_feature_cols:
            transformation = 'Missing category handling and one-hot encoding using training levels only.'
        else:
            transformation = 'No modeling transformation; field excluded before preprocessing.'

        column_audit_rows.append({
            'original_oregon_column': original_name,
            'normalized_column': normalized_name,
            'modeling_name': modeling_name,
            'role': role,
            'status': status,
            'transformation': transformation,
            'reason': reason,
        })

    preprocessing_column_audit = pd.DataFrame(column_audit_rows)
    preprocessing_column_audit.to_csv(output_folder / 'preprocessing_column_audit.csv', index=False)

    preprocessing_audit_summary = pd.DataFrame([{
        'raw_rows': raw_row_count,
        'raw_columns': len(df_raw.columns),
        'unique_members': unique_member_count,
        'repeated_member_episodes_retained': duplicate_member_count,
        'exact_duplicate_rows': exact_duplicate_row_count,
        'missing_outcome_filled_with_zero': missing_outcome_before_fill,
        'positive_outcome_counts_binarized_to_one': positive_counts_binarized,
        'contradictory_treatment_rows': int((df['treatment_derivation_status'] == 'contradictory').sum()),
        'unresolved_treatment_rows': int((df['treatment_derivation_status'] == 'unresolved').sum()),
        'modeling_rows': len(model_df),
        'treated_rows': int((model_df['intervention_flag'] == 1).sum()),
        'control_rows': int((model_df['intervention_flag'] == 0).sum()),
        'outcome_positive_rows': int((model_df['outcome_ed_90d'] == 1).sum()),
        'outcome_zero_rows': int((model_df['outcome_ed_90d'] == 0).sum()),
        'death_related_fields_found': '; '.join(death_related_fields_found),
        'rows_with_available_death_information': rows_with_available_death_information,
        'rows_excluded_for_death': 0,
        'missing_source_risk_tier_rows': missing_risk_tier_rows,
        'out_of_range_source_risk_tier_rows': invalid_risk_tier_rows,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_unique_members': train_df['member_id'].nunique(),
        'test_unique_members': test_df['member_id'].nunique(),
        'train_test_member_overlap': len(set(train_df['member_id']).intersection(test_df['member_id'])),
        'retained_predictor_count': len(feature_cols),
        'encoded_feature_count': combined_matrix.shape[1],
        'zero_variance_predictors_removed': '; '.join(zero_variance_predictors),
        'retained_predictors': '; '.join(feature_cols),
        'post_treatment_fields_found': '; '.join(found_post_treatment_fields),
        'treatment_definition': 'Engaged_flag=1 treated; Engaged_flag=0 control; validated as inverse of OptOut_flag',
        'treatment_distribution': json.dumps(model_df['intervention_flag'].value_counts().sort_index().to_dict()),
        'binary_outcome_distribution': json.dumps(model_df['outcome_ed_90d'].value_counts().sort_index().to_dict()),
        'raw_count_distribution_before_binary': json.dumps(raw_outcome_count_distribution),
        'preprocessing_fit_on_training_only': True,
        'grouped_split_by_member_id': True,
        'leakage_assertions_passed': True,
        'xgboost_execution_backend': 'CUDA GPU required on device 0; modeling grid unchanged',
    }])
    preprocessing_audit_summary.to_csv(output_folder / 'preprocessing_audit_summary.csv', index=False)

    print('Zero-variance predictors removed using training data only:', zero_variance_predictors)
    print('Preprocessing audit summary:')
    display(preprocessing_audit_summary)
    print('Column mapping and role audit:')
    display(preprocessing_column_audit)
    print('Leakage checks passed; final encoded feature count:', combined_matrix.shape[1])
    """,
)

set_cell(
    notebook,
    29,
    """
    def is_binary_indicator_column(series):
        non_missing = pd.Series(series).dropna()
        if non_missing.empty:
            return False
        numeric_values = pd.to_numeric(non_missing, errors='coerce')
        if numeric_values.isna().any():
            return False
        return set(numeric_values.unique()).issubset({0, 1}) and numeric_values.nunique() <= 2


    model_feature_type_rows = []
    for column in feature_cols:
        series = model_df[column]
        if is_binary_indicator_column(series):
            predictor_type = 'Binary indicator'
        elif column in possible_numeric_cols or pd.api.types.is_numeric_dtype(series):
            predictor_type = 'Continuous/count numeric'
        else:
            predictor_type = 'Multi-level categorical'
        model_feature_type_rows.append({'variable': column, 'predictor_type': predictor_type})

    model_feature_type_summary = pd.DataFrame(model_feature_type_rows)
    predictor_type_counts = model_feature_type_summary['predictor_type'].value_counts().to_dict()

    continuous_count_predictors = int(predictor_type_counts.get('Continuous/count numeric', 0))
    binary_indicator_predictors = int(predictor_type_counts.get('Binary indicator', 0))
    multilevel_categorical_predictors = int(predictor_type_counts.get('Multi-level categorical', 0))

    data_review_summary = pd.DataFrame([{
        'total_analytical_records': len(model_df),
        'unique_members': model_df['member_id'].nunique(),
        'treated_records': int((model_df['intervention_flag'] == 1).sum()),
        'control_records': int((model_df['intervention_flag'] == 0).sum()),
        'treatment_rate': float(model_df['intervention_flag'].mean()),
        'outcome_events': int((model_df['outcome_ed_90d'] == 1).sum()),
        'outcome_prevalence': float(model_df['outcome_ed_90d'].mean()),
        'treated_outcome_rate': float(model_df.loc[model_df['intervention_flag'] == 1, 'outcome_ed_90d'].mean()),
        'control_outcome_rate': float(model_df.loc[model_df['intervention_flag'] == 0, 'outcome_ed_90d'].mean()),
        'number_of_predictors': len(feature_cols),
        'number_of_continuous_count_predictors': continuous_count_predictors,
        'number_of_binary_indicator_predictors': binary_indicator_predictors,
        'number_of_multilevel_categorical_predictors': multilevel_categorical_predictors,
        'number_of_model_matrix_columns': x_test.shape[1],
    }])

    data_review_summary_path = output_folder / 'data_review_summary.csv'
    data_review_summary.to_csv(data_review_summary_path, index=False)

    print('Data review summary:')
    display(data_review_summary)
    print('Data review summary written to:', data_review_summary_path)
    """,
)

# Remove the Funds-only death predictor from the explanatory lookup. It is not in
# the Oregon candidate list, but removing the text prevents misleading documentation.
cell31 = "".join(notebook["cells"][31]["source"])
cell31 = cell31.replace("    'death_within_90d_flag': 'Mortality',\n", "")
cell31 = cell31.replace(
    "    'death_within_90d_flag': 'Binary indicator derived from a populated death date; raw death date excluded.',\n",
    "",
)
cell31 = cell31.replace(
    "    numeric_values = pd.to_numeric(raw_series, errors='coerce')\n"
    "    is_numeric_like = pd.api.types.is_numeric_dtype(model_series) or numeric_values.notna().sum() > 0\n",
    "    numeric_values = parse_currency(raw_series) if column == 'total_cost_last_6m' else pd.to_numeric(raw_series, errors='coerce')\n"
    "    is_numeric_like = pd.api.types.is_numeric_dtype(model_series) or numeric_values.notna().sum() > 0\n",
)
cell31 = cell31.replace(
    "    numeric_values = pd.to_numeric(raw_series, errors='coerce')\n"
    "    numeric_share = numeric_values.notna().mean() if len(raw_series) else 0\n",
    "    numeric_values = parse_currency(raw_series) if column == 'total_cost_last_6m' else pd.to_numeric(raw_series, errors='coerce')\n"
    "    numeric_share = numeric_values.notna().mean() if len(raw_series) else 0\n",
)
set_cell(notebook, 31, cell31)

cell33 = "".join(notebook["cells"][33]["source"])
cell33 = cell33.replace(
    "    values = pd.to_numeric(df[column], errors='coerce').dropna()\n",
    "    values = (parse_currency(df[column]) if column == 'total_cost_last_6m' else pd.to_numeric(df[column], errors='coerce')).dropna()\n",
)
set_cell(notebook, 33, cell33)

set_cell(
    notebook,
    67,
    """
    full_x_df = model_df[feature_cols].copy()
    full_prepared_df = prepare_predictor_frame(full_x_df)
    full_matrix = design_matrix_from_prepared(full_prepared_df, combined_matrix.columns)

    full_pred_treated_xgboost = model_treated.predict(make_dmatrix(full_matrix))
    full_pred_control_xgboost = model_control.predict(make_dmatrix(full_matrix))
    scored_full_xgboost = build_uplift_results(model_df, full_pred_treated_xgboost, full_pred_control_xgboost)

    # Backward-compatible alias for the primary XGBoost scored file.
    scored_full = scored_full_xgboost

    if enet_treated is not None and enet_control is not None:
        full_pred_treated_glmnet = enet_treated['best_model'].predict_proba(full_matrix)[:, 1]
        full_pred_control_glmnet = enet_control['best_model'].predict_proba(full_matrix)[:, 1]
        scored_full_glmnet = build_uplift_results(model_df, full_pred_treated_glmnet, full_pred_control_glmnet)
    else:
        full_pred_treated_glmnet = None
        full_pred_control_glmnet = None
        scored_full_glmnet = None
    """,
)

# Cell 91 and the parity manifest inherit Oregon labels from the global replacement.
# Record the SageMaker execution requirement explicitly in the notebook metadata.
notebook.setdefault("metadata", {})["oregon_implementation"] = {
    "source_template": "Code/Uplift Model Code_Funds_Combined.ipynb",
    "source_data": "DataSets/Oregon_2YearDataset.csv",
    "output_root": "Outputs/Uplift_Oregon/Python",
    "treatment_definition": "Engaged_flag=1 treated; Engaged_flag=0 control; OptOut_flag validation only",
    "preprocessing": "member-grouped split and training-only imputation/encoding",
    "death_handling": "no death fields present; no death-derived features; no death-based exclusions",
    "execution_requirement": "SageMaker CUDA GPU required on device 0; grid and model choices unchanged",
}
notebook["metadata"].setdefault("language_info", {})["version"] = "3"

TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {TARGET}")
