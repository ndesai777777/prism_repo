"""Build the Funds Combined uplift notebook from the maintained PRP workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Code" / "Uplift Model Code_rh06032026.ipynb"
TARGET = ROOT / "Code" / "Uplift Model Code_Funds_Combined.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def set_cell(notebook: dict, index: int, text: str) -> None:
    notebook["cells"][index]["source"] = source(text)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


def replace_or_die(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find expected text for {label}")
    return text.replace(old, new)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

cell2 = "".join(notebook["cells"][2]["source"])
cell2 = cell2.replace("    'openpyxl': 'openpyxl',\n", "")
set_cell(notebook, 2, cell2)

markdown_replacements = {
    "TEST AUC FOR CV-TUNED XGBOOST MODELS": "TEST COUNT METRICS FOR CV-TUNED XGBOOST MODELS",
    "OPTIONAL TEST AUC FOR CPU-ONLY GLMNET COMPARISON": "OPTIONAL TEST COUNT METRICS FOR CPU-ONLY ELASTIC-NET COMPARISON",
    "WRITE-UP PROBABILITY CALIBRATION AND BRIER SCORES": "WRITE-UP COUNT CALIBRATION AND REGRESSION ERROR SCORES",
}
for cell in notebook["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    markdown = "".join(cell.get("source", []))
    for old, new in markdown_replacements.items():
        markdown = markdown.replace(old, new)
    cell["source"] = markdown.splitlines(keepends=True)

set_cell(
    notebook,
    0,
    """
    # FUNDS COMBINED UPLIFT MODEL (T-LEARNER AND X-LEARNER)

    Input: `DataSets/Funds_combined.csv`  
    Count outcome: `outcome_ed_90d` (missing values are zero; positive counts are preserved)  
    Treatment: derived jointly from source `intervention_flag` and `OptOut_flag`

    This notebook mirrors the PRP uplift workflow while using leakage-safe Funds-specific
    preprocessing and regression-appropriate evaluation for a nonnegative ED-visit count.
    """,
)

set_cell(
    notebook,
    4,
    """
    from itertools import product
    import json
    import re
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import xgboost as xgb

    try:
        import shap
    except ImportError:
        shap = None

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
    from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from _prism_model_utils import (
        align_to_columns,
        assert_xgb_booster_uses_cuda,
        clean_names_simple,
        ensure_output_folder,
        impute_categorical,
        impute_numeric,
        make_design_matrix,
        ntile_desc,
        project_root,
        require_columns,
        resolve_xgb_gpu_params,
        shap_importance_frame,
        split_train_test,
        to_binary,
        xgb_importance_frame,
        xgb_training_params,
    )

    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    PROJECT_ROOT = project_root()
    """,
)

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

    print('Funds data path:', funds_csv_path, '\\n')
    print('Project root resolved to:', PROJECT_ROOT, '\\n')
    print('Funds output root:', output_root, '\\n')
    print('XGBoost outputs will be saved to:', xgboost_output_folder, '\\n')
    print('GLMNET outputs will be saved to:', glmnet_output_folder, '\\n')
    """,
)

set_cell(
    notebook,
    8,
    """
    def present_columns(columns, frame):
        return [column for column in columns if column in frame.columns]


    def canonical_column_name(value):
        return re.sub(r'[^a-z0-9]+', '', str(value).lower())


    def count_regression_metrics(y_true, y_pred, label):
        actual = np.asarray(y_true, dtype=float)
        predicted = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'mae': float(mean_absolute_error(actual, predicted)),
            'mean_residual': float(np.mean(actual - predicted)),
            'poisson_deviance': float(mean_poisson_deviance(actual, np.clip(predicted, 1e-9, None))),
        }
        print(
            f"{label}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"mean residual={metrics['mean_residual']:.4f}, Poisson deviance={metrics['poisson_deviance']:.4f}"
        )
        return metrics


    REQUIRE_GPU_FOR_XGBOOST = True
    RUN_CPU_ONLY_COMPARISON_MODELS = True
    XGBOOST_CUDA_DEVICE = 0
    XGB_GPU_PARAMS = resolve_xgb_gpu_params(cuda_device=XGBOOST_CUDA_DEVICE) if REQUIRE_GPU_FOR_XGBOOST else {}
    print('XGBoost GPU required:', REQUIRE_GPU_FOR_XGBOOST)
    print('XGBoost CUDA params used for training:', XGB_GPU_PARAMS)
    print('CPU-only elastic-net count comparison enabled:', RUN_CPU_ONLY_COMPARISON_MODELS)


    def make_dmatrix(x_matrix, y=None):
        if y is None:
            return xgb.DMatrix(x_matrix, feature_names=list(x_matrix.columns))
        return xgb.DMatrix(x_matrix, label=np.asarray(y, dtype=float), feature_names=list(x_matrix.columns))


    def fit_xgb_cv_grid(x_matrix, y, grid, nrounds_max=500, nfold=5, seed=123):
        y_array = np.asarray(y, dtype=float)
        if np.any(y_array < 0):
            raise ValueError('Count outcome must be nonnegative for Poisson XGBoost.')
        dtrain = make_dmatrix(x_matrix, y_array)
        folds = int(min(nfold, len(y_array)))
        if folds < 2:
            raise ValueError('Need at least two rows for XGBoost cross-validation.')

        results = []
        best_model_info = None
        best_rmse = np.inf
        for params_grid in grid:
            params_i = xgb_training_params(
                XGB_GPU_PARAMS,
                {
                    'objective': 'count:poisson',
                    'max_delta_step': 0.7,
                    'max_depth': params_grid['max_depth'],
                    'eta': params_grid['eta'],
                    'min_child_weight': params_grid['min_child_weight'],
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                },
                eval_metric='rmse',
                seed=seed,
            )
            cv_i = xgb.cv(
                params=params_i,
                dtrain=dtrain,
                num_boost_round=nrounds_max,
                nfold=folds,
                stratified=False,
                early_stopping_rounds=20,
                seed=seed,
                verbose_eval=False,
            )
            metric_column = 'test-rmse-mean'
            best_iter_i = int(cv_i[metric_column].idxmin())
            best_rmse_i = float(cv_i.loc[best_iter_i, metric_column])
            best_nrounds_i = best_iter_i + 1
            results.append({
                'max_depth': params_grid['max_depth'],
                'eta': params_grid['eta'],
                'min_child_weight': params_grid['min_child_weight'],
                'best_nrounds': best_nrounds_i,
                'cv_rmse': best_rmse_i,
            })
            if best_rmse_i < best_rmse:
                best_rmse = best_rmse_i
                best_model_info = {'params': params_i, 'best_nrounds': best_nrounds_i, 'cv_rmse': best_rmse_i}

        search_results = pd.DataFrame(results).sort_values('cv_rmse').reset_index(drop=True)
        final_model = xgb.train(
            params=best_model_info['params'],
            dtrain=dtrain,
            num_boost_round=best_model_info['best_nrounds'],
            verbose_eval=False,
        )
        return {
            'model': final_model,
            'best_params': best_model_info['params'],
            'best_nrounds': best_model_info['best_nrounds'],
            'best_cv_rmse': best_model_info['cv_rmse'],
            'search_results': search_results,
        }


    class PrefitScaledOutcomePipeline:
        def __init__(self, scaler, model):
            self.named_steps = {'standardscaler': scaler, 'elasticnetcv': model}

        def predict(self, x_matrix):
            return self.named_steps['elasticnetcv'].predict(
                self.named_steps['standardscaler'].transform(x_matrix)
            )


    def fit_elastic_net(x_matrix, y, alpha_grid=np.round(np.arange(0, 1.01, 0.1), 1), nfolds=5, seed=123, prefit_scaler=None):
        y_array = np.asarray(y, dtype=float)
        folds = int(min(nfolds, len(y_array)))
        if folds < 2:
            raise ValueError('Need at least two rows for elastic-net count regression CV.')
        if not RUN_CPU_ONLY_COMPARISON_MODELS:
            raise RuntimeError('Set RUN_CPU_ONLY_COMPARISON_MODELS=True to run elastic-net comparison models.')

        scaler = prefit_scaler if prefit_scaler is not None else StandardScaler().fit(x_matrix)
        x_scaled = scaler.transform(x_matrix)
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
        results = []
        best_pipeline = None
        best_rmse = np.inf
        best_alpha = np.nan
        best_lambda = np.nan

        for alpha in alpha_grid:
            model = ElasticNetCV(
                l1_ratio=float(alpha),
                alphas=np.logspace(-4, 4, 30),
                cv=cv,
                max_iter=10000,
                random_state=seed,
            )
            model.fit(x_scaled, y_array)
            cv_rmse = float(np.sqrt(np.nanmin(np.nanmean(model.mse_path_, axis=1))))
            results.append({'alpha': float(alpha), 'lambda': float(model.alpha_), 'cv_rmse': cv_rmse})
            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_alpha = float(alpha)
                best_lambda = float(model.alpha_)
                best_pipeline = PrefitScaledOutcomePipeline(scaler, model)

        return {
            'best_model': best_pipeline,
            'best_alpha': best_alpha,
            'best_lambda': best_lambda,
            'best_cv_rmse': best_rmse,
            'search_results': pd.DataFrame(results).sort_values('cv_rmse').reset_index(drop=True),
        }


    def build_uplift_results(base_df, pred_treated, pred_control):
        results = base_df.copy()
        results['pred_ed_if_treated'] = np.clip(np.asarray(pred_treated, dtype=float), 0, None)
        results['pred_ed_if_control'] = np.clip(np.asarray(pred_control, dtype=float), 0, None)
        results['benefit_score'] = results['pred_ed_if_control'] - results['pred_ed_if_treated']
        results['uplift_bad_outcome'] = results['pred_ed_if_treated'] - results['pred_ed_if_control']
        results['uplift_decile'] = ntile_desc(results['benefit_score'], 10).to_numpy()
        return results


    def summarize_uplift_deciles(results):
        return (
            results.groupby('uplift_decile', as_index=False)
            .agg(
                n=('outcome_ed_90d', 'size'),
                avg_benefit_score=('benefit_score', 'mean'),
                observed_ed_rate=('outcome_ed_90d', 'mean'),
                treated_pct=('intervention_flag', 'mean'),
                avg_pred_ed_if_treated=('pred_ed_if_treated', 'mean'),
                avg_pred_ed_if_control=('pred_ed_if_control', 'mean'),
            )
            .sort_values('uplift_decile')
        )


    def print_highest_benefit(results, label, n=20):
        print(f'{label} top {n} highest-benefit observations:')
        display(
            results.sort_values('benefit_score', ascending=False)[[
                'member_id', 'outcome_ed_90d', 'intervention_flag',
                'pred_ed_if_treated', 'pred_ed_if_control', 'benefit_score', 'uplift_decile',
            ]].head(n)
        )
        print()


    def glmnet_contribution_importance_frame(model_info, x_matrix, label):
        pipeline = model_info['best_model']
        scaler = pipeline.named_steps['standardscaler']
        regression_model = pipeline.named_steps['elasticnetcv']
        x_scaled = scaler.transform(x_matrix)
        coefficients = regression_model.coef_.ravel()
        contributions = x_scaled * coefficients
        return (
            pd.DataFrame({
                'feature': list(x_matrix.columns),
                'mean_abs_model_contribution': np.abs(contributions).mean(axis=0),
                'coefficient': coefficients,
                'model': label,
                'importance_type': 'standardized_count_regression_contribution',
            })
            .sort_values('mean_abs_model_contribution', ascending=False)
            .reset_index(drop=True)
        )
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
    print('\\nOriginal-to-normalized columns:')
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
        [resolved_treated, resolved_control],
        [1.0, 0.0],
        default=np.nan,
    )
    df['treatment_derivation_status'] = np.select(
        [contradictory_treatment, resolved_treated, resolved_control],
        ['contradictory', 'treated', 'control'],
        default='unresolved',
    )

    outcome_numeric = pd.to_numeric(df['outcome_ed_90d'], errors='coerce')
    missing_outcome_before_fill = int(outcome_numeric.isna().sum())
    if (outcome_numeric.dropna() < 0).any():
        raise ValueError('outcome_ed_90d contains negative counts.')
    df['outcome_ed_90d'] = outcome_numeric.fillna(0.0).astype(float)

    death_text = df['date_of_death'].astype('string').str.strip()
    df['death_within_90d_flag'] = (death_text.notna() & death_text.ne('')).astype(float)

    risk_numeric = pd.to_numeric(df.get('risk_tier'), errors='coerce')
    if risk_numeric.notna().any():
        risk_map = {0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High', 5: 'Very High'}
        df['risk_tier'] = risk_numeric.map(risk_map).fillna('Missing')

    print('Treatment derivation cross-tab:')
    display(pd.crosstab(
        df['source_intervention_flag'].fillna('Missing'),
        df['optout_flag'].fillna('Missing'),
        dropna=False,
    ))
    print('Treatment derivation status:')
    display(df['treatment_derivation_status'].value_counts(dropna=False).rename_axis('status').to_frame('n'))
    print('Missing outcomes filled with zero:', missing_outcome_before_fill)
    print('Duplicate member_id observations retained:', duplicate_member_count)
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
        if canonical in prohibited_canonical or column.startswith('outcome_') and column != 'outcome_ed_90d':
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
    model_df['outcome_stratum'] = np.select(
        [model_df['outcome_ed_90d'].eq(0), model_df['outcome_ed_90d'].eq(1)],
        ['0', '1'],
        default='2_plus',
    )

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
        if not pd.api.types.is_numeric_dtype(model_df[column]):
            model_df[column] = impute_categorical(model_df[column])
        else:
            model_df[column] = impute_numeric(model_df[column])

    predictor_unique_counts = model_df[candidate_predictors].nunique(dropna=True)
    zero_variance_predictors = predictor_unique_counts[predictor_unique_counts <= 1].index.tolist()
    retained_predictors = [column for column in candidate_predictors if column not in zero_variance_predictors]
    for column in zero_variance_predictors:
        exclusion_reasons[column] = 'zero-variance predictor removed after preprocessing'
    candidate_predictors = retained_predictors
    model_df = model_df[[
        'member_id', 'source_intervention_flag', 'optout_flag', 'treatment_derivation_status',
        'outcome_stratum', 'outcome_ed_90d', 'intervention_flag', *candidate_predictors,
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
        stratify_columns=['intervention_flag', 'outcome_stratum'],
    )

    print('Training rows:', len(train_df))
    print('Testing rows:', len(test_df))
    display(
        model_df.groupby(['intervention_flag', 'outcome_stratum'], dropna=False)
        .size().rename('n').reset_index()
    )
    """,
)

set_cell(
    notebook,
    27,
    """
    id_cols = [
        'member_id', 'source_intervention_flag', 'optout_flag',
        'treatment_derivation_status', 'outcome_stratum',
    ]
    feature_cols = list(candidate_predictors)

    train_treated_x_df = train_treated[feature_cols].copy()
    train_control_x_df = train_control[feature_cols].copy()
    test_x_df = test_df[feature_cols].copy()

    combined_matrix, split_matrices = make_design_matrix([train_treated_x_df, train_control_x_df, test_x_df])
    x_treated, x_control, x_test = split_matrices
    y_treated = train_treated['outcome_ed_90d'].astype(float).to_numpy()
    y_control = train_control['outcome_ed_90d'].astype(float).to_numpy()

    forbidden_feature_canonical = prohibited_canonical | raw_treatment_canonical | {
        canonical_column_name('intervention_flag'),
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
    assert 'member_id' not in feature_cols
    assert model_df.loc[model_df['death_within_90d_flag'] == 1].shape[0] > 0

    split_distribution = pd.concat([
        train_df.assign(split='train'), test_df.assign(split='test')
    ]).groupby(['split', 'intervention_flag', 'outcome_stratum'], dropna=False).size().rename('n').reset_index()
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
        'contradictory_treatment_rows': int((df['treatment_derivation_status'] == 'contradictory').sum()),
        'unresolved_treatment_rows': int((df['treatment_derivation_status'] == 'unresolved').sum()),
        'modeling_rows': len(model_df),
        'treated_rows': int((model_df['intervention_flag'] == 1).sum()),
        'control_rows': int((model_df['intervention_flag'] == 0).sum()),
        'death_flag_rows_retained': int(model_df['death_within_90d_flag'].sum()),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'retained_predictor_count': len(feature_cols),
        'encoded_feature_count': combined_matrix.shape[1],
        'zero_variance_predictors_removed': '; '.join(zero_variance_predictors),
        'retained_predictors': '; '.join(feature_cols),
        'treatment_distribution': json.dumps(model_df['intervention_flag'].value_counts().sort_index().to_dict()),
        'outcome_count_distribution': json.dumps(model_df['outcome_ed_90d'].value_counts().sort_index().to_dict()),
        'leakage_assertions_passed': True,
    }])
    preprocessing_audit_summary.to_csv(output_folder / 'preprocessing_audit_summary.csv', index=False)

    print('Preprocessing audit summary:')
    display(preprocessing_audit_summary)
    print('Leakage checks passed; final encoded feature count:', combined_matrix.shape[1])
    """,
)

cell29 = ''.join(notebook['cells'][29]['source'])
cell29 = cell29.replace("'outcome_events': int((model_df['outcome_ed_90d'] == 1).sum()),", "'total_outcome_ed_visits': float(model_df['outcome_ed_90d'].sum()),\n            'members_with_outcome_ed_visit': int((model_df['outcome_ed_90d'] > 0).sum()),")
cell29 = cell29.replace("'outcome_prevalence': float(model_df['outcome_ed_90d'].mean()),", "'mean_outcome_ed_visits': float(model_df['outcome_ed_90d'].mean()),")
cell29 = cell29.replace("'treated_outcome_rate'", "'treated_mean_outcome_ed_visits'")
cell29 = cell29.replace("'control_outcome_rate'", "'control_mean_outcome_ed_visits'")
set_cell(notebook, 29, cell29)

cell31 = ''.join(notebook['cells'][31]['source'])
cell31 = cell31.replace("'dual_eligible': 'Demographics',", "'dual_eligible': 'Demographics',\n    'dual_elg': 'Demographics',\n    'state': 'Demographics',")
cell31 = cell31.replace("'behavioral_health_risk_flag': 'Clinical Conditions',", "'behavioral_health_risk_flag': 'Clinical Conditions',\n    'bh_flag': 'Clinical Conditions',\n    'htn_flag': 'Clinical Conditions',\n    'hyperlipidemia_flag': 'Clinical Conditions',\n    'cad_flag': 'Clinical Conditions',")
cell31 = cell31.replace("'utilities_insecurity_flag': 'SDOH',", "'utilities_insecurity_flag': 'SDOH',\n    'financial_strain_flag': 'SDOH',\n    'healthliteracy_challenges_flag': 'SDOH',")
cell31 = cell31.replace("'pcp_visits_last_6m': 'Utilization',", "'pcp_visits_last_6m': 'Utilization',\n    'op_visits_last_6m': 'Utilization',")
cell31 = cell31.replace("'current_risk_score': 'Risk Scores',", "'current_risk_score': 'Risk Scores',\n    'percolator_score': 'Risk Scores',\n    'death_within_90d_flag': 'Mortality',")
cell31 = cell31.replace("'dual_eligible': 'Indicator or category for Medicare/Medicaid dual eligibility.',", "'dual_eligible': 'Indicator or category for Medicare/Medicaid dual eligibility.',\n    'dual_elg': 'Indicator for Medicare/Medicaid dual eligibility.',\n    'state': 'Member state of residence.',")
cell31 = cell31.replace("'behavioral_health_risk_flag': 'Indicator for behavioral health risk.',", "'behavioral_health_risk_flag': 'Indicator for behavioral health risk.',\n    'bh_flag': 'Indicator for behavioral health history or risk.',\n    'htn_flag': 'Indicator for hypertension history.',\n    'hyperlipidemia_flag': 'Indicator for hyperlipidemia history.',\n    'cad_flag': 'Indicator for coronary artery disease history.',")
cell31 = cell31.replace("'utilities_insecurity_flag': 'Indicator for utility insecurity.',", "'utilities_insecurity_flag': 'Indicator for utility insecurity.',\n    'financial_strain_flag': 'Indicator for financial strain.',\n    'healthliteracy_challenges_flag': 'Indicator for health-literacy challenges.',")
cell31 = cell31.replace("'pcp_visits_last_6m': 'Number of primary care visits in the last 6 months.',", "'pcp_visits_last_6m': 'Number of primary care visits in the last 6 months.',\n    'op_visits_last_6m': 'Number of outpatient visits in the last 6 months.',")
cell31 = cell31.replace("'current_risk_score': 'Current overall risk score.',", "'current_risk_score': 'Current overall risk score.',\n    'percolator_score': 'Baseline Percolator risk score.',\n    'death_within_90d_flag': 'Binary indicator derived from a populated death date; raw death date excluded.',")
set_cell(notebook, 31, cell31)

set_cell(
    notebook,
    35,
    """
    print('Treated count outcome distribution:')
    print(pd.Series(y_treated).value_counts().sort_index())
    print('\\nControl count outcome distribution:')
    print(pd.Series(y_control).value_counts().sort_index())
    if np.any(y_treated < 0) or np.any(y_control < 0):
        raise ValueError('Count outcomes must be nonnegative.')

    dtrain_treated = make_dmatrix(x_treated, y_treated)
    dtrain_control = make_dmatrix(x_control, y_control)
    params = xgb_training_params(
        XGB_GPU_PARAMS,
        {
            'objective': 'count:poisson', 'max_delta_step': 0.7,
            'max_depth': 4, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        eval_metric='rmse', seed=123,
    )
    model_treated = xgb.train(params=params, dtrain=dtrain_treated, num_boost_round=150, verbose_eval=False)
    model_control = xgb.train(params=params, dtrain=dtrain_control, num_boost_round=150, verbose_eval=False)
    assert_xgb_booster_uses_cuda(model_treated, 'Baseline treated XGBoost count model')
    assert_xgb_booster_uses_cuda(model_control, 'Baseline control XGBoost count model')

    test_treated_pos = np.where(test_df['intervention_flag'].to_numpy() == 1)[0]
    test_control_pos = np.where(test_df['intervention_flag'].to_numpy() == 0)[0]
    baseline_treated_metrics = count_regression_metrics(
        test_df['outcome_ed_90d'].iloc[test_treated_pos],
        model_treated.predict(make_dmatrix(x_test.iloc[test_treated_pos])),
        'XGBoost treated baseline count model',
    )
    baseline_control_metrics = count_regression_metrics(
        test_df['outcome_ed_90d'].iloc[test_control_pos],
        model_control.predict(make_dmatrix(x_test.iloc[test_control_pos])),
        'XGBoost control baseline count model',
    )
    """,
)

set_cell(
    notebook,
    39,
    """
    xgb_treated_cv = fit_xgb_cv_grid(x_matrix=x_treated, y=y_treated, grid=xgb_grid, nrounds_max=500, nfold=5)
    model_treated = xgb_treated_cv['model']
    assert_xgb_booster_uses_cuda(model_treated, 'CV-tuned treated XGBoost count model')
    print('XGBoost treated best CV RMSE:', round(xgb_treated_cv['best_cv_rmse'], 4))
    print('XGBoost treated best nrounds:', xgb_treated_cv['best_nrounds'])
    print('XGBoost treated best params:')
    print(xgb_treated_cv['best_params'])
    """,
)

set_cell(
    notebook,
    41,
    """
    xgb_control_cv = fit_xgb_cv_grid(x_matrix=x_control, y=y_control, grid=xgb_grid, nrounds_max=500, nfold=5)
    model_control = xgb_control_cv['model']
    assert_xgb_booster_uses_cuda(model_control, 'CV-tuned control XGBoost count model')
    print('XGBoost control best CV RMSE:', round(xgb_control_cv['best_cv_rmse'], 4))
    print('XGBoost control best nrounds:', xgb_control_cv['best_nrounds'])
    print('XGBoost control best params:')
    print(xgb_control_cv['best_params'])
    """,
)

set_cell(
    notebook,
    43,
    """
    pred_treated_cv_xgb = model_treated.predict(make_dmatrix(x_test.iloc[test_treated_pos]))
    xgb_treated_test_metrics = count_regression_metrics(
        test_df['outcome_ed_90d'].iloc[test_treated_pos], pred_treated_cv_xgb,
        'XGBoost treated CV-tuned test',
    )
    pred_control_cv_xgb = model_control.predict(make_dmatrix(x_test.iloc[test_control_pos]))
    xgb_control_test_metrics = count_regression_metrics(
        test_df['outcome_ed_90d'].iloc[test_control_pos], pred_control_cv_xgb,
        'XGBoost control CV-tuned test',
    )
    print('CV-tuned XGBoost count models trained successfully.')
    """,
)

cell45 = ''.join(notebook['cells'][45]['source'])
cell45 = cell45.replace('GLMNET comparison models', 'elastic-net count comparison models')
cell45 = cell45.replace('sklearn LogisticRegressionCV', 'sklearn ElasticNetCV')
cell45 = cell45.replace("'best_auc'", "'best_cv_rmse'").replace('CV AUC', 'CV RMSE')
set_cell(notebook, 45, cell45)

set_cell(
    notebook,
    47,
    """
    if enet_treated is not None and enet_control is not None:
        pred_treated_cv_glmnet = np.clip(enet_treated['best_model'].predict(x_test.iloc[test_treated_pos]), 0, None)
        glmnet_treated_test_metrics = count_regression_metrics(
            test_df['outcome_ed_90d'].iloc[test_treated_pos], pred_treated_cv_glmnet,
            'Elastic-net treated CV-tuned test',
        )
        pred_control_cv_glmnet = np.clip(enet_control['best_model'].predict(x_test.iloc[test_control_pos]), 0, None)
        glmnet_control_test_metrics = count_regression_metrics(
            test_df['outcome_ed_90d'].iloc[test_control_pos], pred_control_cv_glmnet,
            'Elastic-net control CV-tuned test',
        )
    else:
        pred_treated_cv_glmnet = None
        pred_control_cv_glmnet = None
        glmnet_treated_test_metrics = None
        glmnet_control_test_metrics = None
        print('Skipped elastic-net test metrics because training was skipped.')
    """,
)

cell49 = ''.join(notebook['cells'][49]['source'])
cell49 = cell49.replace("enet_treated['best_model'].predict_proba(x_test)[:, 1]", "np.clip(enet_treated['best_model'].predict(x_test), 0, None)")
cell49 = cell49.replace("enet_control['best_model'].predict_proba(x_test)[:, 1]", "np.clip(enet_control['best_model'].predict(x_test), 0, None)")
set_cell(notebook, 49, cell49)

cell53 = ''.join(notebook['cells'][53]['source'])
cell53 = cell53.replace("enet_control['best_model'].predict_proba(x_treated)[:, 1]", "np.clip(enet_control['best_model'].predict(x_treated), 0, None)")
cell53 = cell53.replace("enet_treated['best_model'].predict_proba(x_control)[:, 1]", "np.clip(enet_treated['best_model'].predict(x_control), 0, None)")
set_cell(notebook, 53, cell53)

set_cell(
    notebook,
    55,
    """
    def count_error_scores_for_results(results, model_label):
        if results is None:
            return None
        rows = []
        for group_label, treatment_value, pred_col in [
            ('Treated', 1, 'pred_ed_if_treated'), ('Control', 0, 'pred_ed_if_control')
        ]:
            subgroup = results[results['intervention_flag'] == treatment_value].copy()
            if subgroup.empty:
                continue
            metrics = count_regression_metrics(subgroup['outcome_ed_90d'], subgroup[pred_col], f'{model_label} {group_label}')
            rows.append({
                'model': model_label, 'group': group_label, 'n': len(subgroup),
                'observed_mean_ed_visits': subgroup['outcome_ed_90d'].mean(),
                'avg_predicted_ed_visits': subgroup[pred_col].mean(), **metrics,
            })
        return pd.DataFrame(rows)


    def calibration_tables_for_results(results, model_label):
        if results is None:
            return None, None
        calibration_parts = []
        for group_label, treatment_value, pred_col in [
            ('Treated', 1, 'pred_ed_if_treated'), ('Control', 0, 'pred_ed_if_control')
        ]:
            subgroup = results[results['intervention_flag'] == treatment_value].copy()
            if subgroup.empty:
                continue
            subgroup['pred_risk_decile'] = ntile_desc(subgroup[pred_col], 10).to_numpy()
            by_decile = subgroup.groupby('pred_risk_decile', as_index=False).agg(
                n=('outcome_ed_90d', 'size'),
                avg_predicted_ed_rate=(pred_col, 'mean'),
                observed_ed_rate=('outcome_ed_90d', 'mean'),
            ).sort_values('pred_risk_decile')
            by_decile.insert(0, 'group', group_label)
            by_decile.insert(0, 'model', model_label)
            by_decile['calibration_error'] = by_decile['observed_ed_rate'] - by_decile['avg_predicted_ed_rate']
            by_decile['abs_calibration_error'] = by_decile['calibration_error'].abs()
            calibration_parts.append(by_decile)
        if not calibration_parts:
            return None, None
        calibration_by_decile = pd.concat(calibration_parts, ignore_index=True)
        rows = []
        for (model_name, group_label), frame in calibration_by_decile.groupby(['model', 'group']):
            rows.append({
                'model': model_name, 'group': group_label, 'n': frame['n'].sum(),
                'mean_abs_calibration_error': frame['abs_calibration_error'].mean(),
                'weighted_mean_abs_calibration_error': np.average(frame['abs_calibration_error'], weights=frame['n']),
                'max_abs_calibration_error': frame['abs_calibration_error'].max(),
            })
        return calibration_by_decile, pd.DataFrame(rows)


    def save_calibration_plot(calibration_by_decile, folder, model_label):
        if calibration_by_decile is None or calibration_by_decile.empty:
            return
        groups = [group for group in ['Control', 'Treated'] if group in set(calibration_by_decile['group'])]
        fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), sharey=True)
        if len(groups) == 1:
            axes = [axes]
        max_rate = max(calibration_by_decile['avg_predicted_ed_rate'].max(), calibration_by_decile['observed_ed_rate'].max(), 0.01)
        for ax, group_label in zip(axes, groups):
            group_df = calibration_by_decile[calibration_by_decile['group'] == group_label].sort_values('pred_risk_decile')
            x_values = np.arange(len(group_df)); bar_width = 0.38
            ax.bar(x_values - bar_width / 2, group_df['avg_predicted_ed_rate'], width=bar_width, label='Avg predicted ED visits', color='#4C78A8')
            ax.bar(x_values + bar_width / 2, group_df['observed_ed_rate'], width=bar_width, label='Observed mean ED visits', color='#F58518')
            ax.set_title(f'{model_label}: {group_label} Count Calibration')
            ax.set_xlabel('Predicted Count Decile: 1 = Highest')
            ax.set_xticks(x_values); ax.set_xticklabels(group_df['pred_risk_decile'].astype(str))
            ax.set_ylim(0, max_rate * 1.22); ax.grid(axis='y', alpha=0.25)
        axes[0].set_ylabel('Mean 90-Day ED Visits')
        axes[-1].legend(loc='upper right')
        fig.suptitle(f'{model_label}: Predicted vs Observed Mean ED Visits by Decile', y=1.03)
        fig.tight_layout(); fig.savefig(folder / 'dashboard_calibration_plot.png', dpi=150, bbox_inches='tight'); plt.close(fig)


    def save_count_evaluation_outputs(results, folder, model_label):
        error_df = count_error_scores_for_results(results, model_label)
        calibration_by_decile, calibration_summary = calibration_tables_for_results(results, model_label)
        if error_df is not None:
            error_df.to_csv(folder / 'model_count_error_scores.csv', index=False)
            display(error_df)
        if calibration_by_decile is not None:
            calibration_by_decile.to_csv(folder / 'calibration_by_decile.csv', index=False)
            calibration_summary.to_csv(folder / 'calibration_summary.csv', index=False)
            save_calibration_plot(calibration_by_decile, folder, model_label)
            display(calibration_summary)
        return error_df, calibration_by_decile, calibration_summary


    count_errors_xgboost, calibration_by_decile_xgboost, calibration_summary_xgboost = save_count_evaluation_outputs(
        results_test_xgboost, xgboost_output_folder, 'XGBoost'
    )
    if results_test_glmnet is not None:
        count_errors_glmnet, calibration_by_decile_glmnet, calibration_summary_glmnet = save_count_evaluation_outputs(
            results_test_glmnet, glmnet_output_folder, 'GLMNET'
        )
    else:
        count_errors_glmnet = calibration_by_decile_glmnet = calibration_summary_glmnet = None
    """,
)

set_cell(
    notebook,
    57,
    """
    def factual_event_count_summary(train_df, test_df):
        rows = []
        for split_name, frame in [('Train', train_df), ('Test', test_df)]:
            for group_label, treatment_value in [('Treated', 1), ('Control', 0)]:
                subgroup = frame[frame['intervention_flag'] == treatment_value]
                rows.append({
                    'split': split_name, 'group': group_label, 'n': len(subgroup),
                    'total_ed_visits': float(subgroup['outcome_ed_90d'].sum()),
                    'members_with_ed_visit': int((subgroup['outcome_ed_90d'] > 0).sum()),
                    'member_event_rate': float((subgroup['outcome_ed_90d'] > 0).mean()) if len(subgroup) else np.nan,
                    'mean_ed_visits': float(subgroup['outcome_ed_90d'].mean()) if len(subgroup) else np.nan,
                })
        return pd.DataFrame(rows)


    def factual_prediction_diagnostics(results, model_label):
        diagnostic_rows = []
        range_rows = []
        for group_label, treatment_value, pred_col in [
            ('Treated', 1, 'pred_ed_if_treated'), ('Control', 0, 'pred_ed_if_control')
        ]:
            subgroup = results[results['intervention_flag'] == treatment_value].copy()
            actual = subgroup['outcome_ed_90d'].astype(float)
            predicted = np.clip(subgroup[pred_col].astype(float), 0, None)
            metrics = count_regression_metrics(actual, predicted, f'{model_label} {group_label}')
            diagnostic_rows.append({
                'model': model_label, 'group': group_label, 'prediction_column': pred_col,
                'n': len(subgroup), 'observed_mean_ed_visits': actual.mean(),
                'predicted_mean_ed_visits': predicted.mean(),
                'pearson_actual_predicted': actual.corr(predicted, method='pearson'),
                'spearman_actual_predicted': actual.corr(predicted, method='spearman'), **metrics,
            })
            range_rows.append({
                'model': model_label, 'group': group_label, 'prediction_column': pred_col,
                'n': len(subgroup), 'min_pred': predicted.min(), 'p10_pred': predicted.quantile(.1),
                'median_pred': predicted.median(), 'mean_pred': predicted.mean(),
                'p90_pred': predicted.quantile(.9), 'max_pred': predicted.max(),
            })
        return pd.DataFrame(diagnostic_rows), pd.DataFrame(range_rows)


    event_count_summary = factual_event_count_summary(train_df, test_df)
    event_count_summary.to_csv(output_folder / 'factual_event_count_summary.csv', index=False)
    diagnostic_frames = []; range_frames = []
    sep_xgboost, range_xgboost = factual_prediction_diagnostics(results_test_xgboost, 'XGBoost')
    diagnostic_frames.append(sep_xgboost); range_frames.append(range_xgboost)
    if results_test_glmnet is not None:
        sep_glmnet, range_glmnet = factual_prediction_diagnostics(results_test_glmnet, 'GLMNET')
        diagnostic_frames.append(sep_glmnet); range_frames.append(range_glmnet)
    factual_prediction_separation = pd.concat(diagnostic_frames, ignore_index=True)
    factual_prediction_ranges = pd.concat(range_frames, ignore_index=True)
    factual_prediction_separation.to_csv(output_folder / 'factual_prediction_separation.csv', index=False)
    factual_prediction_ranges.to_csv(output_folder / 'factual_prediction_ranges.csv', index=False)
    display(event_count_summary); display(factual_prediction_separation); display(factual_prediction_ranges)
    """,
)

cell59 = ''.join(notebook['cells'][59]['source'])
pattern = re.compile(r"def wilson_score_interval\(.*?\n\ndef observed_gap_by_decile", re.S)
replacement = dedent("""
def difference_in_means_ci(control_values, treated_values, confidence=0.95):
    control = pd.Series(control_values, dtype=float).dropna()
    treated = pd.Series(treated_values, dtype=float).dropna()
    if control.empty or treated.empty:
        return np.nan, np.nan, np.nan
    gap = float(control.mean() - treated.mean())
    se = np.sqrt(control.var(ddof=1) / len(control) + treated.var(ddof=1) / len(treated))
    z_value = 1.959964 if confidence == 0.95 else 1.959964
    return gap, gap - z_value * se, gap + z_value * se


def observed_gap_by_decile""")
cell59, count = pattern.subn(replacement, cell59, count=1)
if count != 1:
    raise RuntimeError('Could not replace binary confidence-interval helpers in cell 59')
old_call = """gap, gap_ci_lower, gap_ci_upper = difference_in_proportions_ci(
            control_events,
            len(control),
            treated_events,
            len(treated),
        )"""
new_call = """gap, gap_ci_lower, gap_ci_upper = difference_in_means_ci(
            control['outcome_ed_90d'], treated['outcome_ed_90d']
        )"""
cell59 = replace_or_die(cell59, old_call, new_call, 'first count gap call')
cell59 = cell59.replace("'ci_method': 'newcombe_wilson_difference_in_proportions'", "'ci_method': 'normal_approximation_difference_in_mean_counts'")
cell59 = cell59.replace('Observed ED Rate', 'Observed Mean ED Visits').replace('ED Rate', 'Mean ED Visits')
set_cell(notebook, 59, cell59)

cell63 = ''.join(notebook['cells'][63]['source'])
cell63 = cell63.replace("enet_treated['best_model'].predict_proba(full_matrix)[:, 1]", "np.clip(enet_treated['best_model'].predict(full_matrix), 0, None)")
cell63 = cell63.replace("enet_control['best_model'].predict_proba(full_matrix)[:, 1]", "np.clip(enet_control['best_model'].predict(full_matrix), 0, None)")
set_cell(notebook, 63, cell63)

set_cell(
    notebook,
    67,
    """
    print('INTERPRETATION:')
    print('- pred_ed_if_treated = predicted mean 90-day ED-visit count if treated')
    print('- pred_ed_if_control = predicted mean 90-day ED-visit count if not treated')
    print('- benefit_score = pred_ed_if_control - pred_ed_if_treated')
    print('- Higher benefit_score means treatment is predicted to prevent more ED visits')
    print('- Uplift decile 1 = highest predicted treatment benefit')
    """,
)

cell69 = ''.join(notebook['cells'][69]['source'])
cell69 = cell69.replace('Observed 90-Day ED Rate', 'Observed Mean 90-Day ED Visits')
cell69 = cell69.replace("'Observed ED Rate'", "'Mean ED Visits'")
cell69 = cell69.replace('Predicted ED Risk', 'Predicted Mean ED Visits')
set_cell(notebook, 69, cell69)

cell75 = ''.join(notebook['cells'][75]['source'])
cell75 = cell75.replace('standardized logit-contribution', 'standardized count-regression contribution')
cell75 = cell75.replace('Mean Absolute Standardized Logit Contribution', 'Mean Absolute Standardized Count-Regression Contribution')
set_cell(notebook, 75, cell75)

cell77 = ''.join(notebook['cells'][77]['source'])
cell77 = cell77.replace("named_steps['logisticregressioncv']", "named_steps['elasticnetcv']")
cell77 = cell77.replace('GLMNET logit scale', 'GLMNET standardized count scale')
cell77 = cell77.replace('glmnet_standardized_logit_contribution_difference', 'glmnet_standardized_count_contribution_difference')
cell77 = cell77.replace('Mean Absolute Standardized Logit Contribution to Benefit', 'Mean Absolute Standardized Count Contribution to Benefit')
set_cell(notebook, 77, cell77)

cell81 = ''.join(notebook['cells'][81]['source'])
cell81 = cell81.replace("treated_glmnet_pipeline.predict_proba(frame)[:, 1]", "np.clip(treated_glmnet_pipeline.predict(frame), 0, None)")
cell81 = cell81.replace("control_glmnet_pipeline.predict_proba(frame)[:, 1]", "np.clip(control_glmnet_pipeline.predict(frame), 0, None)")
set_cell(notebook, 81, cell81)

set_cell(
    notebook,
    83,
    """
    def get_group_metric(frame, group, metric_col):
        if frame is None or metric_col not in frame.columns:
            return np.nan
        match = frame[frame['group'] == group]
        return float(match[metric_col].iloc[0]) if not match.empty else np.nan


    def get_top_gap_metric(gap_df, metric_col):
        if gap_df is None or gap_df.empty:
            return np.nan
        match = gap_df[gap_df['uplift_decile'] == 1]
        return match[metric_col].iloc[0] if not match.empty else np.nan


    def get_curve_metric(curve_summary_df, metric_col):
        if curve_summary_df is None or curve_summary_df.empty:
            return np.nan
        return float(curve_summary_df[metric_col].iloc[0])


    def decile_gap_rank_correlation(gap_df):
        valid = gap_df[['avg_predicted_benefit', 'observed_control_minus_treated_gap']].dropna()
        return valid['avg_predicted_benefit'].corr(valid['observed_control_minus_treated_gap'], method='spearman') if len(valid) >= 2 else np.nan


    def build_model_evaluation_row(model_label, treated_cv_rmse, control_cv_rmse, treated_test_metrics, control_test_metrics, error_df, calibration_summary, gap_df, curve_summary_df):
        return {
            'model': model_label,
            'treated_cv_rmse': treated_cv_rmse, 'control_cv_rmse': control_cv_rmse,
            'treated_test_rmse': treated_test_metrics['rmse'], 'control_test_rmse': control_test_metrics['rmse'],
            'treated_test_mae': treated_test_metrics['mae'], 'control_test_mae': control_test_metrics['mae'],
            'treated_test_poisson_deviance': treated_test_metrics['poisson_deviance'],
            'control_test_poisson_deviance': control_test_metrics['poisson_deviance'],
            'treated_calibration_error': get_group_metric(calibration_summary, 'Treated', 'weighted_mean_abs_calibration_error'),
            'control_calibration_error': get_group_metric(calibration_summary, 'Control', 'weighted_mean_abs_calibration_error'),
            'top_decile_avg_predicted_benefit': get_top_gap_metric(gap_df, 'avg_predicted_benefit'),
            'top_decile_observed_control_minus_treated_gap': get_top_gap_metric(gap_df, 'observed_control_minus_treated_gap'),
            'top_decile_observed_gap_ci_lower_95': get_top_gap_metric(gap_df, 'observed_gap_ci_lower_95'),
            'top_decile_observed_gap_ci_upper_95': get_top_gap_metric(gap_df, 'observed_gap_ci_upper_95'),
            'benefit_gap_spearman_corr_by_decile': decile_gap_rank_correlation(gap_df),
            'observed_gap_area_under_curve': get_curve_metric(curve_summary_df, 'observed_gap_area_under_curve'),
            'predicted_benefit_area_under_curve': get_curve_metric(curve_summary_df, 'predicted_benefit_area_under_curve'),
            'auc_applicability': 'not_applicable_for_count_outcome',
            'brier_applicability': 'not_applicable_for_count_outcome',
        }


    model_evaluation_rows = [build_model_evaluation_row(
        'XGBoost', xgb_treated_cv['best_cv_rmse'], xgb_control_cv['best_cv_rmse'],
        xgb_treated_test_metrics, xgb_control_test_metrics, count_errors_xgboost,
        calibration_summary_xgboost, observed_gap_xgboost, uplift_curve_summary_xgboost,
    )]
    if enet_treated is not None and enet_control is not None:
        model_evaluation_rows.append(build_model_evaluation_row(
            'GLMNET', enet_treated['best_cv_rmse'], enet_control['best_cv_rmse'],
            glmnet_treated_test_metrics, glmnet_control_test_metrics, count_errors_glmnet,
            calibration_summary_glmnet, observed_gap_glmnet, uplift_curve_summary_glmnet,
        ))
    model_evaluation_summary = pd.DataFrame(model_evaluation_rows)
    model_evaluation_summary.to_csv(output_folder / 'model_evaluation_summary.csv', index=False)
    display(model_evaluation_summary)
    """,
)

set_cell(
    notebook,
    85,
    """
    recommendation_rows = []
    for _, row in model_evaluation_summary.iterrows():
        avg_rmse = np.nanmean([row['treated_test_rmse'], row['control_test_rmse']])
        avg_mae = np.nanmean([row['treated_test_mae'], row['control_test_mae']])
        avg_calibration = np.nanmean([row['treated_calibration_error'], row['control_calibration_error']])
        recommendation_rows.append({
            'model': row['model'],
            'count_prediction_performance': f'Average test RMSE {avg_rmse:.3f}; average test MAE {avg_mae:.3f}',
            'count_calibration': f'Average weighted absolute calibration error {avg_calibration:.3f}',
            'benefit_ranking_quality': f"Top-decile predicted benefit {row['top_decile_avg_predicted_benefit']:.3f}; decile rank correlation {row['benefit_gap_spearman_corr_by_decile']:.3f}",
            'explainability': 'Tree SHAP risk and benefit drivers' if row['model'] == 'XGBoost' else 'Standardized elastic-net contribution and permutation SHAP drivers',
            'model_evaluation_takeaway': 'Evaluate count prediction, calibration, observed mean-count gaps, uncertainty, and ROI jointly before operational use.',
        })
    model_recommendation_summary = pd.DataFrame(recommendation_rows)
    model_recommendation_summary.to_csv(output_folder / 'model_recommendation_summary.csv', index=False)
    display(model_recommendation_summary)
    """,
)

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

cell91 = ''.join(notebook['cells'][91]['source'])
cell91 = cell91.replace("output_folder / 'data_review_summary.csv',", "output_folder / 'data_review_summary.csv',\n    output_folder / 'preprocessing_audit_summary.csv',\n    output_folder / 'preprocessing_column_audit.csv',\n    output_folder / 'preprocessing_split_distribution.csv',")
cell91 = cell91.replace("folder / 'model_brier_scores.csv',", "folder / 'model_count_error_scores.csv',")
set_cell(notebook, 91, cell91)

set_cell(
    notebook,
    97,
    """
    # Build an explicit PRP-to-Funds output parity manifest without modifying PRP outputs.
    prp_python_root = PROJECT_ROOT / 'Outputs' / 'Uplift' / 'Python'
    incidental_parts = {'.ipynb_checkpoints', '__pycache__'}
    synthetic_tokens = ('true_benefit', 'true_driver')
    brier_filename = 'model_brier_scores.csv'

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
        elif prp_path.name == brier_filename:
            applicability = 'not_applicable'
            explanation = 'Brier score is not valid for a nonbinary count outcome; model_count_error_scores.csv is generated instead.'
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
                'explanation': 'Funds-specific preprocessing, count-outcome, or schema artifact.',
            })

    output_parity_manifest = pd.DataFrame(parity_rows)
    output_parity_manifest.to_csv(output_folder / 'output_parity_manifest.csv', index=False)
    missing_applicable = output_parity_manifest.query("applicability_status == 'applicable' and generated_status == False")
    print('Output parity status counts:')
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
