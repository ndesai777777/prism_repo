"""Rebuild the Funds notebook from the PRP notebook with binary Funds preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Code" / "Uplift Model Code_rh06032026.ipynb"
TARGET = ROOT / "Code" / "Uplift Model Code_Funds_Combined.ipynb"


def lines(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def set_cell(notebook: dict, index: int, text: str) -> None:
    notebook["cells"][index]["source"] = lines(text)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

set_cell(
    notebook,
    0,
    """
    # FUNDS COMBINED UPLIFT MODEL (T-LEARNER WITH XGBOOST)

    Input: `DataSets/Funds_combined.csv`  
    Binary outcome: `outcome_ed_90d` (1 = one or more ED visits; missing = 0)  
    Treatment: derived jointly from source `intervention_flag` and `OptOut_flag`

    This notebook preserves the PRP uplift notebook's modeling, tuning, diagnostics,
    calibration, Brier scoring, charts, and output methods. Only Funds-specific data
    loading, preprocessing, leakage controls, and output isolation differ.
    """,
)

cell2 = "".join(notebook["cells"][2]["source"])
cell2 = cell2.replace("    'openpyxl': 'openpyxl',\n", "")
set_cell(notebook, 2, cell2)

cell4 = "".join(notebook["cells"][4]["source"])
cell4 = cell4.replace(
    "from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV\n",
    "from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, LogisticRegressionCV\n",
)
set_cell(notebook, 4, cell4)

cell8 = "".join(notebook["cells"][8]["source"])
old_fit_elastic_net = '''def fit_elastic_net(x_matrix, y, alpha_grid=np.round(np.arange(0, 1.01, 0.1), 1), nfolds=5, seed=123, prefit_scaler=None):
    y_array = np.asarray(y, dtype=float)
    class_counts = pd.Series(y_array).value_counts()
    folds = int(min(nfolds, class_counts.min())) if len(class_counts) > 1 else 0
    if folds < 2:
        raise ValueError('Need at least two outcome classes with at least two rows each for elastic-net CV.')

    if not RUN_CPU_ONLY_COMPARISON_MODELS:
        raise RuntimeError(
            'fit_elastic_net uses sklearn LogisticRegressionCV, which trains on CPU. '
            'Set RUN_CPU_ONLY_COMPARISON_MODELS = True to run this CPU comparison model.'
        )

    results = []
    best_pipeline = None
    best_auc = -np.inf
    best_alpha = np.nan
    best_lambda = np.nan
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for alpha in alpha_grid:
        penalty = 'l2' if alpha == 0 else 'elasticnet'
        l1_ratios = None if alpha == 0 else [float(alpha)]
        cv_model = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 30),
            cv=cv,
            penalty=penalty,
            solver='saga',
            l1_ratios=l1_ratios,
            scoring='roc_auc',
            max_iter=10000,
            random_state=seed,
            refit=True,
        )
        if prefit_scaler is None:
            pipeline = make_pipeline(StandardScaler(), cv_model)
            pipeline.fit(x_matrix, y_array)
            fitted = pipeline.named_steps['logisticregressioncv']
        else:
            x_scaled = prefit_scaler.transform(x_matrix)
            cv_model.fit(x_scaled, y_array)
            fitted = cv_model
            pipeline = PrefitScaledLogisticPipeline(prefit_scaler, fitted)
        scores = fitted.scores_[1.0]
        auc_cv = float(np.nanmax(np.nanmean(scores, axis=0)))
        lambda_value = float(1 / fitted.C_[0])
        results.append({'alpha': float(alpha), 'lambda': lambda_value, 'cv_auc': auc_cv})
        if auc_cv > best_auc:
            best_auc = auc_cv
            best_alpha = float(alpha)
            best_lambda = lambda_value
            best_pipeline = pipeline

    return {
        'best_model': best_pipeline,
        'best_alpha': best_alpha,
        'best_lambda': best_lambda,
        'best_auc': best_auc,
        'search_results': pd.DataFrame(results).sort_values('cv_auc', ascending=False).reset_index(drop=True),
    }
'''
new_fit_elastic_net = '''def fit_elastic_net(x_matrix, y, alpha=0.5, lambda_value=1.0, seed=123, prefit_scaler=None):
    y_array = np.asarray(y, dtype=float)
    if pd.Series(y_array).nunique() < 2:
        raise ValueError('Need both outcome classes to fit the fixed elastic-net model.')

    if not RUN_CPU_ONLY_COMPARISON_MODELS:
        raise RuntimeError(
            'fit_elastic_net uses sklearn LogisticRegression, which trains on CPU. '
            'Set RUN_CPU_ONLY_COMPARISON_MODELS = True to run this CPU comparison model.'
        )

    # Fixed standard elastic-net specification: no CV and no hyperparameter search.
    # sklearn uses C = 1 / lambda, so lambda=1 corresponds to its default C=1.
    model = LogisticRegression(
        C=float(1 / lambda_value),
        penalty='elasticnet',
        solver='saga',
        l1_ratio=float(alpha),
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )

    if prefit_scaler is None:
        scaler = StandardScaler().fit(x_matrix)
    else:
        scaler = prefit_scaler
    model.fit(scaler.transform(x_matrix), y_array)
    pipeline = PrefitScaledLogisticPipeline(scaler, model)
    fixed_specification = pd.DataFrame([{
        'alpha': float(alpha),
        'lambda': float(lambda_value),
        'cv_auc': np.nan,
        'selection_method': 'fixed_no_tuning',
    }])
    return {
        'best_model': pipeline,
        'best_alpha': float(alpha),
        'best_lambda': float(lambda_value),
        'best_auc': np.nan,
        'search_results': fixed_specification,
    }
'''
if old_fit_elastic_net not in cell8:
    raise RuntimeError("Could not locate the source GLMNet tuning function.")
cell8 = cell8.replace(old_fit_elastic_net, new_fit_elastic_net)
set_cell(notebook, 8, cell8)

cell45 = "".join(notebook["cells"][45]["source"])
cell45 = cell45.replace(
    "Training GLMNET comparison models on CPU with sklearn LogisticRegressionCV.",
    "Training fixed GLMNET comparison models on CPU without hyperparameter tuning.",
)
cell45 = cell45.replace("Best treated alpha:", "Fixed treated alpha:")
cell45 = cell45.replace("Best treated lambda:", "Fixed treated lambda:")
cell45 = cell45.replace(
    "    print('Best treated CV AUC:', round(enet_treated['best_auc'], 4))\n",
    "    print('Treated CV AUC: not calculated (hyperparameter tuning disabled)')\n",
)
cell45 = cell45.replace("Best control alpha:", "Fixed control alpha:")
cell45 = cell45.replace("Best control lambda:", "Fixed control lambda:")
cell45 = cell45.replace(
    "    print('Best control CV AUC:', round(enet_control['best_auc'], 4))\n",
    "    print('Control CV AUC: not calculated (hyperparameter tuning disabled)')\n",
)
set_cell(notebook, 45, cell45)

cell53 = "".join(notebook["cells"][53]["source"])
old_xlearner_glmnet_regression = '''def fit_glmnet_regression_model(x_matrix, y, seed=123, prefit_scaler=None):
    y_array = np.asarray(y, dtype=float)
    folds = int(min(5, len(y_array)))
    if folds < 2:
        raise ValueError('Need at least two rows for GLMNET X-learner regression.')
    reg_model = ElasticNetCV(
        l1_ratio=np.round(np.arange(0.0, 1.01, 0.1), 1),
        alphas=np.logspace(-4, 2, 50),
        cv=folds,
        max_iter=10000,
        random_state=seed,
    )
    if prefit_scaler is None:
        pipeline = make_pipeline(StandardScaler(), reg_model)
        pipeline.fit(x_matrix, y_array)
    else:
        x_scaled = prefit_scaler.transform(x_matrix)
        reg_model.fit(x_scaled, y_array)
        pipeline = PrefitScaledRegressionPipeline(prefit_scaler, reg_model)
    return pipeline
'''
new_xlearner_glmnet_regression = '''def fit_glmnet_regression_model(x_matrix, y, seed=123, prefit_scaler=None):
    y_array = np.asarray(y, dtype=float)
    if len(y_array) < 2:
        raise ValueError('Need at least two rows for GLMNET X-learner regression.')
    reg_model = ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )
    if prefit_scaler is None:
        scaler = StandardScaler().fit(x_matrix)
    else:
        scaler = prefit_scaler
    reg_model.fit(scaler.transform(x_matrix), y_array)
    return PrefitScaledRegressionPipeline(scaler, reg_model)
'''
if old_xlearner_glmnet_regression not in cell53:
    raise RuntimeError("Could not locate the source X-learner GLMNet regression function.")
cell53 = cell53.replace(old_xlearner_glmnet_regression, new_xlearner_glmnet_regression)

old_propensity_model = '''def fit_propensity_model(x_matrix, treatment, seed=123):
    treatment_array = np.asarray(treatment, dtype=float)
    class_counts = pd.Series(treatment_array).value_counts()
    folds = int(min(5, class_counts.min())) if len(class_counts) > 1 else 0
    if folds < 2:
        raise ValueError('Need at least two treatment classes with at least two rows each for propensity modeling.')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    propensity_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 30),
        cv=cv,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.5],
        scoring='roc_auc',
        max_iter=10000,
        random_state=seed,
        refit=True,
    )
    pipeline = make_pipeline(StandardScaler(), propensity_model)
    pipeline.fit(x_matrix, treatment_array)
    return pipeline
'''
new_propensity_model = '''def fit_propensity_model(x_matrix, treatment, seed=123):
    treatment_array = np.asarray(treatment, dtype=float)
    if pd.Series(treatment_array).nunique() < 2:
        raise ValueError('Need both treatment classes for propensity modeling.')
    propensity_model = LogisticRegression(
        C=1.0,
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )
    pipeline = make_pipeline(StandardScaler(), propensity_model)
    pipeline.fit(x_matrix, treatment_array)
    return pipeline
'''
if old_propensity_model not in cell53:
    raise RuntimeError("Could not locate the source propensity tuning function.")
cell53 = cell53.replace(old_propensity_model, new_propensity_model)
set_cell(notebook, 53, cell53)

set_cell(
    notebook,
    6,
    """
    funds_csv_path = PROJECT_ROOT / 'DataSets' / 'Funds_combined.csv'
    if not funds_csv_path.exists():
        nested_candidate = PROJECT_ROOT / 'prism_repo' / 'DataSets' / 'Funds_combined.csv'
        if nested_candidate.exists():
            funds_csv_path = nested_candidate
        else:
            raise FileNotFoundError('Could not locate Funds_combined.csv in either expected repository location.')

    output_root = ensure_output_folder(PROJECT_ROOT / 'Outputs' / 'Uplift_Funds')
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

    print('Funds data path:', funds_csv_path)
    print('Project root resolved to:', PROJECT_ROOT)
    print('Funds output root:', output_root)
    print('XGBoost outputs will be saved to:', xgboost_output_folder)
    print('GLMNET outputs will be saved to:', glmnet_output_folder)
    """,
)

set_cell(
    notebook,
    10,
    """
    df_raw = pd.read_csv(funds_csv_path)
    original_column_names = list(df_raw.columns)
    df_raw.columns = clean_names_simple(df_raw.columns)
    if pd.Index(df_raw.columns).duplicated().any():
        duplicates = pd.Index(df_raw.columns)[pd.Index(df_raw.columns).duplicated()].tolist()
        raise ValueError(f'Duplicate columns after normalization: {duplicates}')

    print('Rows:', len(df_raw))
    print('Columns:', len(df_raw.columns))
    print('Original-to-normalized columns:')
    display(pd.DataFrame({'original': original_column_names, 'normalized': df_raw.columns}))
    """,
)

set_cell(
    notebook,
    14,
    """
    df = df_raw.copy()
    raw_row_count = len(df)
    duplicate_member_count = int(df['member_id'].duplicated().sum())

    source_intervention = to_binary(df['intervention_flag'])
    source_optout = to_binary(df['optout_flag'])
    source_intervention = source_intervention.where(source_intervention.isin([0.0, 1.0]))
    source_optout = source_optout.where(source_optout.isin([0.0, 1.0]))

    treated_signal = source_intervention.eq(1.0)
    control_signal = source_optout.eq(1.0)
    contradictory_treatment = treated_signal & control_signal
    resolved_treated = treated_signal & ~control_signal
    resolved_control = control_signal & ~treated_signal

    df['source_intervention_flag'] = source_intervention
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

    death_text = df['date_of_death'].astype('string').str.strip()
    df['death_within_90d_flag'] = (death_text.notna() & death_text.ne('')).astype(float)

    risk_numeric = pd.to_numeric(df['risk_tier'], errors='coerce')
    missing_risk_tier_rows = int(risk_numeric.isna().sum())
    invalid_risk_tier_rows = int((risk_numeric.notna() & ~risk_numeric.isin([1, 2, 3, 4, 5])).sum())
    valid_risk_tier = risk_numeric.where(risk_numeric.isin([1, 2, 3, 4, 5]))
    # Preserve the original Funds tier assignments. Values outside 1-5 are unassigned,
    # not recoded from current_risk_score or collapsed into PRP risk labels.
    df['risk_tier'] = valid_risk_tier.astype('Int64').astype('string').fillna('Missing')

    print('Treatment derivation cross-tab:')
    display(pd.crosstab(
        df['source_intervention_flag'].fillna('Missing'),
        df['optout_flag'].fillna('Missing'),
        dropna=False,
    ))
    print('Treatment derivation status:')
    display(df['treatment_derivation_status'].value_counts(dropna=False).rename_axis('status').to_frame('n'))
    print('Missing outcomes filled with zero:', missing_outcome_before_fill)
    print('Members with one or more ED visits after binarization:', positive_counts_binarized)
    print('Duplicate member_id observations retained:', duplicate_member_count)
    print('Original Funds risk-tier distribution (1 = lowest, 5 = highest):')
    display(df['risk_tier'].value_counts(dropna=False).reindex(['1', '2', '3', '4', '5', 'Missing'], fill_value=0))
    print('Missing source risk tiers:', missing_risk_tier_rows)
    print('Out-of-range source risk tiers treated as unassigned:', invalid_risk_tier_rows)
    print('Death flag distribution:')
    display(df['death_within_90d_flag'].value_counts(dropna=False).sort_index())
    """,
)

set_cell(
    notebook,
    16,
    """
    # No features are derived from program, engagement, intervention, or outcome dates.
    # These are post-treatment fields for the Funds analysis and are excluded before modeling.
    print('Post-treatment date feature engineering intentionally skipped for Funds Combined.')
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
        'death_within_90d_flag',
    ]
    prohibited_names = [
        'programstart', 'programend', 'engaged_date', 'engagement_length',
        'interventioncount', 'successful_intervention', 'successfulinterventions',
        'outcome_ed_30d', 'outcome_ed_6mo', 'outcome_admit_30d', 'outcome_admit_90d',
        'outcome_admit_6mo', 'outcome_total_cost_90d', 'outcome_total_cost_6mo',
        'total_cost_6mo', 'date_of_death', 'case_manager_name',
    ]
    prohibited_canonical = {canonical_column_name(column) for column in prohibited_names}
    raw_treatment_canonical = {
        canonical_column_name('source_intervention_flag'),
        canonical_column_name('optout_flag'),
    }

    candidate_predictors = [column for column in candidate_predictors_all if column in df.columns]
    missing_predictors = [column for column in candidate_predictors_all if column not in df.columns]
    exclusion_reasons = {}
    for column in df.columns:
        canonical = canonical_column_name(column)
        if column in candidate_predictors:
            continue
        if canonical in prohibited_canonical or (column.startswith('outcome_') and column != 'outcome_ed_90d'):
            exclusion_reasons[column] = 'post-treatment, alternate outcome, or explicitly prohibited'
        elif canonical in raw_treatment_canonical or column == 'intervention_flag':
            exclusion_reasons[column] = 'treatment definition/source; retained as metadata but excluded from predictors'
        elif column == 'member_id':
            exclusion_reasons[column] = 'identifier retained in outputs but excluded from predictors'
        elif column == 'treatment_derivation_status':
            exclusion_reasons[column] = 'treatment derivation audit metadata'
        else:
            exclusion_reasons[column] = 'not on leakage-safe baseline predictor allowlist'

    meta_columns = [
        'member_id', 'source_intervention_flag', 'optout_flag',
        'treatment_derivation_status', 'outcome_ed_90d', 'intervention_flag',
    ]
    model_df = df[[*meta_columns, *candidate_predictors]].copy()
    unresolved_mask = ~model_df['treatment_derivation_status'].isin(['treated', 'control'])
    excluded_treatment_rows = int(unresolved_mask.sum())
    model_df = model_df.loc[~unresolved_mask].copy().reset_index(drop=True)

    if missing_predictors:
        print('Allowlisted predictors not found:', missing_predictors)
    print('Rows excluded because treatment could not be resolved:', excluded_treatment_rows)
    print('Candidate baseline predictors:', candidate_predictors)
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
        'death_within_90d_flag',
    ]
    for column in present_columns(binary_predictors, model_df):
        model_df[column] = to_binary(model_df[column])

    possible_numeric_cols = [
        'age', 'op_visits_last_6m', 'ed_visits_last_30d', 'ed_visits_last_6m',
        'admits_last_6m', 'observation_stays_last_6m', 'total_cost_last_6m',
        'rx_count_last_6m', 'current_risk_score', 'percolator_score',
    ]
    for column in present_columns(possible_numeric_cols, model_df):
        model_df[column] = pd.to_numeric(model_df[column], errors='coerce')

    for column in candidate_predictors:
        if pd.api.types.is_numeric_dtype(model_df[column]):
            model_df[column] = impute_numeric(model_df[column])
        else:
            model_df[column] = impute_categorical(model_df[column])

    predictor_unique_counts = model_df[candidate_predictors].nunique(dropna=True)
    zero_variance_predictors = predictor_unique_counts[predictor_unique_counts <= 1].index.tolist()
    retained_predictors = [column for column in candidate_predictors if column not in zero_variance_predictors]
    for column in zero_variance_predictors:
        exclusion_reasons[column] = 'zero-variance predictor removed after preprocessing'
    candidate_predictors = retained_predictors
    model_df = model_df[[
        'member_id', 'source_intervention_flag', 'optout_flag', 'treatment_derivation_status',
        'outcome_ed_90d', 'intervention_flag', *candidate_predictors,
    ]].copy()

    print('Zero-variance predictors removed:', zero_variance_predictors)
    print('Final modeling columns:')
    print(list(model_df.columns))
    """,
)

set_cell(
    notebook,
    23,
    """
    train_df, test_df = split_train_test(
        model_df,
        train_fraction=0.70,
        seed=123,
        stratify_columns=['intervention_flag', 'outcome_ed_90d'],
    )

    print('Training rows:', len(train_df))
    print('Testing rows:', len(test_df))
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

    id_cols = ['member_id', 'source_intervention_flag', 'optout_flag', 'treatment_derivation_status']
    feature_cols = list(candidate_predictors)

    train_treated_x_df = train_treated[feature_cols].copy()
    train_control_x_df = train_control[feature_cols].copy()
    test_x_df = test_df[feature_cols].copy()
    combined_matrix, split_matrices = make_design_matrix([train_treated_x_df, train_control_x_df, test_x_df])
    x_treated, x_control, x_test = split_matrices
    y_treated = train_treated['outcome_ed_90d'].astype(float).to_numpy()
    y_control = train_control['outcome_ed_90d'].astype(float).to_numpy()

    forbidden_feature_canonical = prohibited_canonical | raw_treatment_canonical | {
        canonical_column_name('intervention_flag')
    }
    leaking_features = [
        column for column in feature_cols
        if any(canonical_column_name(column).startswith(item) for item in forbidden_feature_canonical)
    ]
    leaking_matrix_columns = [
        column for column in combined_matrix.columns
        if any(canonical_column_name(column).startswith(item) for item in forbidden_feature_canonical)
    ]
    assert not leaking_features, f'Prohibited predictors found: {leaking_features}'
    assert not leaking_matrix_columns, f'Encoded leakage columns found: {leaking_matrix_columns}'
    assert set(pd.Series(y_treated).unique()).issubset({0.0, 1.0})
    assert set(pd.Series(y_control).unique()).issubset({0.0, 1.0})
    assert 'member_id' not in feature_cols
    assert model_df.loc[model_df['death_within_90d_flag'] == 1].shape[0] > 0

    split_distribution = pd.concat([
        train_df.assign(split='train'), test_df.assign(split='test')
    ]).groupby(['split', 'intervention_flag', 'outcome_ed_90d'], dropna=False).size().rename('n').reset_index()
    split_distribution.to_csv(output_folder / 'preprocessing_split_distribution.csv', index=False)

    column_audit_rows = []
    for column in df.columns:
        if column in feature_cols:
            status, reason = 'included', 'leakage-safe baseline predictor'
        else:
            status, reason = 'excluded', exclusion_reasons.get(column, 'not selected')
        column_audit_rows.append({'column': column, 'status': status, 'reason': reason})
    preprocessing_column_audit = pd.DataFrame(column_audit_rows)
    preprocessing_column_audit.to_csv(output_folder / 'preprocessing_column_audit.csv', index=False)

    preprocessing_audit_summary = pd.DataFrame([{
        'raw_rows': raw_row_count,
        'raw_columns': len(df_raw.columns),
        'duplicate_member_id_observations_retained': duplicate_member_count,
        'missing_outcome_filled_with_zero': missing_outcome_before_fill,
        'positive_outcome_counts_binarized_to_one': positive_counts_binarized,
        'contradictory_treatment_rows': int((df['treatment_derivation_status'] == 'contradictory').sum()),
        'unresolved_treatment_rows': int((df['treatment_derivation_status'] == 'unresolved').sum()),
        'modeling_rows': len(model_df),
        'treated_rows': int((model_df['intervention_flag'] == 1).sum()),
        'control_rows': int((model_df['intervention_flag'] == 0).sum()),
        'outcome_positive_rows': int((model_df['outcome_ed_90d'] == 1).sum()),
        'outcome_zero_rows': int((model_df['outcome_ed_90d'] == 0).sum()),
        'death_flag_rows_retained': int(model_df['death_within_90d_flag'].sum()),
        'missing_source_risk_tier_rows': missing_risk_tier_rows,
        'out_of_range_source_risk_tier_rows': invalid_risk_tier_rows,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'retained_predictor_count': len(feature_cols),
        'encoded_feature_count': combined_matrix.shape[1],
        'zero_variance_predictors_removed': '; '.join(zero_variance_predictors),
        'retained_predictors': '; '.join(feature_cols),
        'treatment_distribution': json.dumps(model_df['intervention_flag'].value_counts().sort_index().to_dict()),
        'binary_outcome_distribution': json.dumps(model_df['outcome_ed_90d'].value_counts().sort_index().to_dict()),
        'raw_count_distribution_before_binary': json.dumps(raw_outcome_count_distribution),
        'leakage_assertions_passed': True,
    }])
    preprocessing_audit_summary.to_csv(output_folder / 'preprocessing_audit_summary.csv', index=False)
    print('Preprocessing audit summary:')
    display(preprocessing_audit_summary)
    print('Leakage checks passed; final encoded feature count:', combined_matrix.shape[1])
    """,
)

cell31 = "".join(notebook["cells"][31]["source"])
replacements = {
    "'dual_eligible': 'Demographics',": "'dual_eligible': 'Demographics',\n    'dual_elg': 'Demographics',\n    'state': 'Demographics',",
    "'behavioral_health_risk_flag': 'Clinical Conditions',": "'behavioral_health_risk_flag': 'Clinical Conditions',\n    'bh_flag': 'Clinical Conditions',\n    'htn_flag': 'Clinical Conditions',\n    'hyperlipidemia_flag': 'Clinical Conditions',\n    'cad_flag': 'Clinical Conditions',",
    "'utilities_insecurity_flag': 'SDOH',": "'utilities_insecurity_flag': 'SDOH',\n    'financial_strain_flag': 'SDOH',\n    'healthliteracy_challenges_flag': 'SDOH',",
    "'pcp_visits_last_6m': 'Utilization',": "'pcp_visits_last_6m': 'Utilization',\n    'op_visits_last_6m': 'Utilization',",
    "'current_risk_score': 'Risk Scores',": "'current_risk_score': 'Risk Scores',\n    'percolator_score': 'Risk Scores',\n    'death_within_90d_flag': 'Mortality',",
    "'dual_eligible': 'Indicator or category for Medicare/Medicaid dual eligibility.',": "'dual_eligible': 'Indicator or category for Medicare/Medicaid dual eligibility.',\n    'dual_elg': 'Indicator for Medicare/Medicaid dual eligibility.',\n    'state': 'Member state of residence.',",
    "'behavioral_health_risk_flag': 'Indicator for behavioral health risk.',": "'behavioral_health_risk_flag': 'Indicator for behavioral health risk.',\n    'bh_flag': 'Indicator for behavioral health history or risk.',\n    'htn_flag': 'Indicator for hypertension history.',\n    'hyperlipidemia_flag': 'Indicator for hyperlipidemia history.',\n    'cad_flag': 'Indicator for coronary artery disease history.',",
    "'utilities_insecurity_flag': 'Indicator for utility insecurity.',": "'utilities_insecurity_flag': 'Indicator for utility insecurity.',\n    'financial_strain_flag': 'Indicator for financial strain.',\n    'healthliteracy_challenges_flag': 'Indicator for health-literacy challenges.',",
    "'pcp_visits_last_6m': 'Number of primary care visits in the last 6 months.',": "'pcp_visits_last_6m': 'Number of primary care visits in the last 6 months.',\n    'op_visits_last_6m': 'Number of outpatient visits in the last 6 months.',",
    "'current_risk_score': 'Current overall risk score.',": "'current_risk_score': 'Current overall risk score.',\n    'percolator_score': 'Baseline Percolator risk score.',\n    'death_within_90d_flag': 'Binary indicator derived from a populated death date; raw death date excluded.',",
}
for old, new in replacements.items():
    if old not in cell31:
        raise RuntimeError(f"Missing expected predictor dictionary text: {old}")
    cell31 = cell31.replace(old, new)
set_cell(notebook, 31, cell31)

set_cell(
    notebook,
    87,
    """
    ---
    ## SYNTHETIC TRUE-BENEFIT VALIDATION — NOT APPLICABLE

    Funds Combined does not provide known counterfactual treatment-effect ground truth.
    Synthetic true-benefit calculations are intentionally not performed or fabricated.
    This exception is recorded in `output_parity_manifest.csv`.
    """,
)
set_cell(
    notebook,
    88,
    """
    synthetic_true_benefit_validation_status = pd.DataFrame([{
        'section': 'synthetic_true_benefit_validation',
        'applicability': 'not_applicable',
        'reason': 'Funds Combined has no known counterfactual treatment-effect ground truth.',
    }])
    display(synthetic_true_benefit_validation_status)
    """,
)

cell89 = "".join(notebook["cells"][89]["source"])
cell89 = cell89.replace(
    "risk_tier_order = ['Low', 'Medium', 'High', 'Very High']\n",
    "risk_tier_order = ['1', '2', '3', '4', '5']\n"
    "risk_tier_display = {\n"
    "    '1': 'Tier 1 (lowest risk)',\n"
    "    '2': 'Tier 2',\n"
    "    '3': 'Tier 3',\n"
    "    '4': 'Tier 4',\n"
    "    '5': 'Tier 5 (highest risk)',\n"
    "}\n",
)
old_thresholds = '''risk_tier_thresholds = pd.DataFrame(
    [
        {
            'risk_tier': 'Low',
            'current_risk_score_rule': '< 35',
            'lower_bound_inclusive': np.nan,
            'upper_bound_exclusive': 35,
        },
        {
            'risk_tier': 'Medium',
            'current_risk_score_rule': '35 to < 55',
            'lower_bound_inclusive': 35,
            'upper_bound_exclusive': 55,
        },
        {
            'risk_tier': 'High',
            'current_risk_score_rule': '55 to < 75',
            'lower_bound_inclusive': 55,
            'upper_bound_exclusive': 75,
        },
        {
            'risk_tier': 'Very High',
            'current_risk_score_rule': '>= 75',
            'lower_bound_inclusive': 75,
            'upper_bound_exclusive': np.nan,
        },
    ]
)
'''
new_thresholds = '''risk_tier_thresholds = pd.DataFrame(
    [
        {
            'risk_tier': tier,
            'risk_tier_label': risk_tier_display[tier],
            'current_risk_score_rule': 'Not derived from current_risk_score',
            'source_definition': 'Original Funds Combined risk_tier assignment',
        }
        for tier in risk_tier_order
    ]
)
'''
if old_thresholds not in cell89:
    raise RuntimeError("Could not locate the PRP risk-tier threshold table.")
cell89 = cell89.replace(old_thresholds, new_thresholds)

old_population_summary = '''risk_tier_population_summary = (
    scored_full_glmnet.groupby('risk_tier', observed=False)
    .agg(
        members=('risk_tier', 'size'),
        pct_population=('risk_tier', lambda s: len(s) / len(scored_full_glmnet)),
        min_current_risk_score=('current_risk_score', 'min'),
        max_current_risk_score=('current_risk_score', 'max'),
        avg_current_risk_score=('current_risk_score', 'mean'),
    )
    .reindex(risk_tier_order)
    .reset_index()
)
'''
new_population_summary = '''valid_risk_tier_population = scored_full_glmnet.loc[
    scored_full_glmnet['risk_tier'].isin(risk_tier_order)
].copy()
unassigned_risk_tier_members = int(len(scored_full_glmnet) - len(valid_risk_tier_population))

risk_tier_population_summary = (
    valid_risk_tier_population.groupby('risk_tier', observed=False)
    .agg(
        members=('risk_tier', 'size'),
        pct_population=(
            'risk_tier',
            lambda s: len(s) / len(valid_risk_tier_population),
        ),
        min_current_risk_score=('current_risk_score', 'min'),
        max_current_risk_score=('current_risk_score', 'max'),
        avg_current_risk_score=('current_risk_score', 'mean'),
    )
    .reindex(risk_tier_order)
    .reset_index()
)
risk_tier_population_summary.insert(
    1,
    'risk_tier_label',
    risk_tier_population_summary['risk_tier'].map(risk_tier_display),
)
'''
if old_population_summary not in cell89:
    raise RuntimeError("Could not locate the PRP risk-tier population summary.")
cell89 = cell89.replace(old_population_summary, new_population_summary)
cell89 = cell89.replace(
    "    df = scored_df.copy()\n\n    df['risk_tier'] = pd.Categorical(\n",
    "    df = scored_df.loc[scored_df['risk_tier'].isin(risk_tier_order)].copy()\n\n"
    "    df['risk_tier'] = pd.Categorical(\n",
)
cell89 = cell89.replace(
    "        [f'{tier} risk\\n(n={counts.loc[tier]})' for tier in risk_tier_order]\n",
    "        [f'{risk_tier_display[tier]}\\n(n={counts.loc[tier]})' for tier in risk_tier_order]\n",
)
cell89 = cell89.replace(
    "ax.set_xlabel('Risk tier based on current_risk_score', labelpad=14)",
    "ax.set_xlabel('Original Funds risk tier (1 = lowest, 5 = highest)', labelpad=14)",
)
cell89 = cell89.replace(
    "print('Risk tier thresholds:')\n",
    "print('Funds risk-tier definitions (original assignments; no current_risk_score derivation):')\n",
)
cell89 = cell89.replace(
    "print('Risk tier population summary:')\n",
    "print('Members excluded from the 1-5 risk-tier population summary because tier was missing or invalid:', unassigned_risk_tier_members)\n"
    "print('Risk tier population summary (original Funds tiers 1-5):')\n",
)
set_cell(notebook, 89, cell89)

cell91 = "".join(notebook["cells"][91]["source"])
cell91 = cell91.replace(
    "output_folder / 'data_review_summary.csv',",
    "output_folder / 'data_review_summary.csv',\n    output_folder / 'preprocessing_audit_summary.csv',\n    output_folder / 'preprocessing_column_audit.csv',\n    output_folder / 'preprocessing_split_distribution.csv',",
)
set_cell(notebook, 91, cell91)

set_cell(
    notebook,
    97,
    """
    # Build an explicit PRP-to-Funds output parity manifest without modifying PRP outputs.
    prp_python_root = PROJECT_ROOT / 'Outputs' / 'Uplift' / 'Python'
    incidental_parts = {'.ipynb_checkpoints', '__pycache__'}
    synthetic_tokens = ('true_benefit', 'true_driver')
    prp_files = [
        path for path in prp_python_root.rglob('*')
        if path.is_file() and not incidental_parts.intersection(path.parts) and path.name != '.gitkeep'
    ]
    parity_rows = []
    expected_funds_paths = set()
    for prp_path in sorted(prp_files):
        relative = prp_path.relative_to(prp_python_root)
        funds_path = output_folder / relative
        relative_text = relative.as_posix()
        applicability = 'applicable'
        explanation = ''
        if any(token in relative_text.lower() for token in synthetic_tokens):
            applicability = 'not_applicable'
            explanation = 'Synthetic counterfactual ground truth is unavailable for Funds Combined.'
        elif relative.parts[:2] in [('Predictor_Distributions', 'Categorical'), ('Predictor_Distributions', 'Numeric')] and not funds_path.exists():
            applicability = 'not_applicable'
            explanation = 'Predictor-specific plot is schema-dependent and this PRP predictor is not retained for Funds.'
        if applicability == 'applicable':
            expected_funds_paths.add(funds_path.resolve())
        parity_rows.append({
            'corresponding_prp_relative_path': relative_text,
            'expected_funds_relative_path': relative_text,
            'artifact_category': relative.parts[0] if len(relative.parts) > 1 else 'Python root',
            'applicability_status': applicability,
            'generated_status': funds_path.exists() if applicability == 'applicable' else False,
            'explanation': explanation,
        })

    existing_funds_files = [
        path for path in output_folder.rglob('*')
        if path.is_file() and path.name != 'output_parity_manifest.csv' and not incidental_parts.intersection(path.parts)
    ]
    for funds_path in sorted(existing_funds_files):
        if funds_path.resolve() not in expected_funds_paths:
            relative = funds_path.relative_to(output_folder).as_posix()
            parity_rows.append({
                'corresponding_prp_relative_path': '',
                'expected_funds_relative_path': relative,
                'artifact_category': 'Funds-specific addition',
                'applicability_status': 'funds_specific',
                'generated_status': True,
                'explanation': 'Funds-specific preprocessing or schema artifact.',
            })

    output_parity_manifest = pd.DataFrame(parity_rows)
    output_parity_manifest.to_csv(output_folder / 'output_parity_manifest.csv', index=False)
    missing_applicable = output_parity_manifest.query("applicability_status == 'applicable' and generated_status == False")
    display(output_parity_manifest.groupby(['applicability_status', 'generated_status']).size().rename('n').reset_index())
    if not missing_applicable.empty:
        print('Applicable PRP counterparts not generated:')
        display(missing_applicable)
    else:
        print('All applicable PRP output counterparts were generated.')

    final_encoded_leaks = [
        column for column in combined_matrix.columns
        if any(canonical_column_name(column).startswith(item) for item in forbidden_feature_canonical)
    ]
    assert not final_encoded_leaks, f'Final leakage audit failed: {final_encoded_leaks}'
    print('Final encoded-feature leakage audit passed.')
    """,
)

notebook["metadata"]["language_info"]["version"] = "3"
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {TARGET}")
