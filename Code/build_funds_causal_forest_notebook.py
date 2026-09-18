"""Build the Funds Combined causal-forest workflow notebook.

The notebook deliberately mirrors the PRISM causal-forest workflow while reusing
the leakage-safe Funds preprocessing implemented in
``Uplift Model Code_Funds_Combined.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Code" / "PRISM_Causal_Forest_Modeling_Funds_Workflow.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        """
        # PRISM Causal Forest Modeling — Funds Combined

        This notebook is the report-ready causal forest workflow for the Funds Combined
        population. It preserves the causal-forest diagnostics, stability-aware tuning,
        treatment-effect scoring, HTE deciles, uplift-framework comparisons,
        explainability, and business-value analyses from
        `PRISM_Causal_Forest_Modeling_Workflow.ipynb`, while using the exact leakage-safe
        preprocessing rules established in `Uplift Model Code_Funds_Combined.ipynb`.

        Core sign convention:

        ```text
        tau_hat = estimated effect of intervention on outcome_ed_90d
        benefit_score = -tau_hat
        higher benefit_score = larger estimated ED risk reduction from intervention
        ```

        Input: `DataSets/Funds_combined.csv`<br>
        Output root: `Outputs/Causal-Forests_Funds/Python`<br>
        Reproducibility seed: `123`
        """
    ),
    markdown(
        """
        ## Optional Package Install

        Run this cell only if the kernel is missing `econml`. The repository's Python
        3.13 environment contains the required packages.
        """
    ),
    code(
        """
        # Uncomment if needed in a compatible Python environment.
        # %pip install econml shap scikit-learn pandas numpy matplotlib
        """
    ),
    markdown(
        """
        ## Background

        Risk identifies who may experience an adverse outcome; causal benefit estimates
        who may improve because of intervention. Funds Combined contains real-world
        treatment and outcome data without known member-level counterfactual truth, so
        this workflow emphasizes overlap, uncertainty, stability, held-out ranking, and
        agreement with the Funds T-learner and X-learner workflows.
        """
    ),
    markdown(
        """
        ## Business Question

        Which Funds Combined members are most likely to benefit from intervention through
        a reduction in 90-day emergency-department utilization?
        """
    ),
    markdown(
        """
        ## Project Objectives

        - Reproduce the Funds treatment and outcome derivations exactly.
        - Use only baseline, leakage-safe predictors.
        - Estimate member-level heterogeneous treatment effects with causal forest.
        - quantify overlap, uncertainty, stability, and held-out HTE separation.
        - Compare causal-forest rankings with the Funds T-learner and X-learner outputs.
        - Produce decision-ready decile, explainability, risk-tier, and savings artifacts.
        """
    ),
    markdown(
        """
        ## Analytical Task 1: Understanding And Explaining The Causal Forest Framework

        The outcome is the binary indicator `outcome_ed_90d`. The treatment is derived
        jointly from the source intervention and opt-out flags. Causal forest uses
        cross-fitting and flexible nuisance models to estimate how the treatment effect
        varies with baseline member characteristics. Treatment, opt-out, identifiers,
        post-treatment process variables, alternate outcomes, and raw dates are excluded
        from the predictor matrix.
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib.util
        import json
        import re
        import sys
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        try:
            from IPython.display import display
        except Exception:
            display = print

        if importlib.util.find_spec('econml') is None:
            raise ImportError(
                'Missing required package: econml. Use the repository Python 3.13 '
                'environment or install econml in a compatible kernel.'
            )
        from econml.dml import CausalForestDML

        CODE_DIR = Path.cwd()
        if CODE_DIR.name.lower() != 'code':
            CODE_DIR = Path.cwd() / 'Code'
        if not (CODE_DIR / '_prism_model_utils.py').exists():
            raise FileNotFoundError('Run this notebook from the repository root or Code folder.')
        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))

        from _prism_model_utils import (
            clean_names_simple,
            ensure_output_folder,
            impute_categorical,
            impute_numeric,
            make_design_matrix,
            ntile_desc,
            require_columns,
            split_train_test,
            to_binary,
        )

        PROJECT_ROOT = CODE_DIR.parent
        FUNDS_CSV_PATH = PROJECT_ROOT / 'DataSets' / 'Funds_combined.csv'
        if not FUNDS_CSV_PATH.exists():
            nested_candidate = PROJECT_ROOT / 'prism_repo' / 'DataSets' / 'Funds_combined.csv'
            if nested_candidate.exists():
                FUNDS_CSV_PATH = nested_candidate
            else:
                raise FileNotFoundError('Could not locate DataSets/Funds_combined.csv.')

        SEED = 123
        TRAIN_FRACTION = 0.70
        OUTCOME_COL = 'outcome_ed_90d'
        TREATMENT_COL = 'intervention_flag'
        ED_VISIT_COST = 1200.0
        INTERVENTION_COST = 250.0
        SHAP_MAX_EXPLAIN_ROWS = 100
        OUTPUT_DIR = ensure_output_folder(PROJECT_ROOT / 'Outputs' / 'Causal-Forests_Funds' / 'Python')

        np.random.seed(SEED)
        warnings.filterwarnings('ignore', category=UserWarning)

        print(f'Funds input: {FUNDS_CSV_PATH}')
        print(f'Output directory: {OUTPUT_DIR}')
        print(f'Seed: {SEED}')
        """
    ),
    code(
        """
        PATHS = {
            'scored_output': OUTPUT_DIR / 'causal_forest_scored_output.csv',
            'test_scored_output': OUTPUT_DIR / 'causal_forest_scored_test_output.csv',
            'data_review_summary': OUTPUT_DIR / 'causal_forest_data_review_summary.csv',
            'preprocessing_audit_summary': OUTPUT_DIR / 'causal_forest_preprocessing_audit_summary.csv',
            'preprocessing_column_audit': OUTPUT_DIR / 'causal_forest_preprocessing_column_audit.csv',
            'split_distribution': OUTPUT_DIR / 'causal_forest_preprocessing_split_distribution.csv',
            'column_name_mapping': OUTPUT_DIR / 'causal_forest_source_column_name_mapping.csv',
            'predictor_inventory': OUTPUT_DIR / 'causal_forest_predictor_inventory.csv',
            'event_count_summary': OUTPUT_DIR / 'causal_forest_event_count_summary.csv',
            'propensity_summary': OUTPUT_DIR / 'causal_forest_propensity_summary.csv',
            'effect_distribution_summary': OUTPUT_DIR / 'causal_forest_effect_distribution_summary.csv',
            'uncertainty_summary': OUTPUT_DIR / 'causal_forest_uncertainty_summary.csv',
            'ate_summary': OUTPUT_DIR / 'causal_forest_ate_summary.csv',
            'hyperparameter_tuning_summary': OUTPUT_DIR / 'causal_forest_hyperparameter_tuning_summary.csv',
            'hyperparameter_stability_pairs': OUTPUT_DIR / 'causal_forest_hyperparameter_stability_pairs.csv',
            'decile_summary': OUTPUT_DIR / 'causal_forest_decile_summary.csv',
            'observed_gap_summary': OUTPUT_DIR / 'causal_forest_observed_gap_by_decile.csv',
            'risk_tier_benefit_group_summary': OUTPUT_DIR / 'causal_forest_risk_tier_benefit_group_summary.csv',
            'top_benefit_examples': OUTPUT_DIR / 'causal_forest_top_benefit_examples.csv',
            'true_benefit_validation_summary': OUTPUT_DIR / 'causal_forest_true_benefit_validation_summary.csv',
            'variable_importance': OUTPUT_DIR / 'causal_forest_variable_importance.csv',
            'top_decile_profile': OUTPUT_DIR / 'causal_forest_top_decile_profile.csv',
            'targeting_summary': OUTPUT_DIR / 'causal_forest_targeting_summary.csv',
            'cumulative_targeting': OUTPUT_DIR / 'causal_forest_cumulative_gross_savings_by_targeting.csv',
            'marginal_targeting': OUTPUT_DIR / 'causal_forest_marginal_gross_savings_by_targeting.csv',
            'marginal_advantage': OUTPUT_DIR / 'causal_forest_marginal_gross_savings_advantage_vs_current_risk.csv',
            'consistency_summary': OUTPUT_DIR / 'causal_forest_vs_uplift_consistency_summary.csv',
            'shap_importance': OUTPUT_DIR / 'causal_forest_global_benefit_shap_importance.csv',
            'shap_member_values': OUTPUT_DIR / 'causal_forest_member_benefit_shap_values.csv',
            'output_index': OUTPUT_DIR / 'causal_forest_output_index.csv',
            'propensity_chart': OUTPUT_DIR / 'dashboard_propensity_overlap.png',
            'effect_distribution_chart': OUTPUT_DIR / 'dashboard_causal_forest_effect_distribution.png',
            'benefit_decile_chart': OUTPUT_DIR / 'dashboard_causal_forest_avg_benefit_by_decile.png',
            'tau_decile_chart': OUTPUT_DIR / 'dashboard_causal_forest_tau_by_decile.png',
            'observed_gap_chart': OUTPUT_DIR / 'dashboard_causal_forest_observed_gap_by_decile.png',
            'risk_tier_benefit_group_chart': OUTPUT_DIR / 'dashboard_causal_forest_risk_tier_by_benefit_group.png',
            'variable_importance_chart': OUTPUT_DIR / 'dashboard_causal_forest_variable_importance.png',
            'shap_chart': OUTPUT_DIR / 'dashboard_causal_forest_global_benefit_shap.png',
            'cumulative_targeting_chart': OUTPUT_DIR / 'dashboard_cumulative_gross_savings_targeting.png',
            'marginal_advantage_chart': OUTPUT_DIR / 'dashboard_marginal_gross_savings_advantage_vs_current_risk.png',
        }

        def save_csv(frame, path):
            frame.to_csv(path, index=False)
            print(f'Saved: {path}')

        def save_current_figure(path):
            plt.tight_layout()
            plt.savefig(path, dpi=180, bbox_inches='tight')
            plt.close()
            print(f'Saved: {path}')

        def canonical_column_name(value):
            return re.sub(r'[^a-z0-9]+', '', str(value).lower())

        def present_columns(columns, frame):
            return [column for column in columns if column in frame.columns]

        def summarize_distribution(values, label):
            series = pd.Series(values, dtype=float).dropna()
            return pd.DataFrame({
                'metric': ['mean', 'std_dev', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'],
                label: [series.mean(), series.std(), series.min(), series.quantile(0.10),
                        series.quantile(0.25), series.median(), series.quantile(0.75),
                        series.quantile(0.90), series.max()],
            })

        def safe_corr(a, b, method):
            joined = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1).dropna()
            return joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method) if len(joined) >= 3 else np.nan

        def top_overlap(a_scores, b_scores, share=0.10):
            a = pd.Series(a_scores).reset_index(drop=True)
            b = pd.Series(b_scores).reset_index(drop=True)
            n = min(len(a), len(b))
            if n == 0:
                return np.nan
            k = max(1, int(np.floor(n * share)))
            return len(set(a.iloc[:n].nlargest(k).index) & set(b.iloc[:n].nlargest(k).index)) / k

        def effect_standard_errors(model, x_matrix):
            try:
                return np.asarray(model.effect_inference(x_matrix).stderr, dtype=float)
            except Exception as exc:
                print(f'Standard errors not available: {exc}')
                return np.full(len(x_matrix), np.nan)

        def fit_funds_propensity_model(x_matrix, treatment, seed=SEED):
            treatment_array = np.asarray(treatment, dtype=float)
            if pd.Series(treatment_array).nunique() < 2:
                raise ValueError('Both treatment classes are required for propensity modeling.')
            # Exact fixed Funds uplift specification: elastic net, alpha=.5, lambda=1 (C=1).
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=0.5,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=seed,
                ),
            ).fit(x_matrix, treatment_array)

        def clipped_propensity(model, x_matrix, lower=0.05, upper=0.95):
            return np.clip(model.predict_proba(x_matrix)[:, 1], lower, upper)

        def propensity_auc(y_true, scores):
            try:
                return roc_auc_score(y_true, scores)
            except Exception:
                return np.nan
        """
    ),
    markdown("## Analytical Task 2: Data Review And Funds-Specific Preprocessing"),
    code(
        """
        df_raw = pd.read_csv(FUNDS_CSV_PATH)
        original_column_names = list(df_raw.columns)
        df_raw.columns = clean_names_simple(df_raw.columns)
        if pd.Index(df_raw.columns).duplicated().any():
            duplicates = pd.Index(df_raw.columns)[pd.Index(df_raw.columns).duplicated()].tolist()
            raise ValueError(f'Duplicate columns after normalization: {duplicates}')
        require_columns(df_raw, ['member_id', OUTCOME_COL, TREATMENT_COL, 'optout_flag', 'date_of_death', 'risk_tier'])

        column_name_mapping = pd.DataFrame({'original': original_column_names, 'normalized': df_raw.columns})
        save_csv(column_name_mapping, PATHS['column_name_mapping'])

        df = df_raw.copy()
        raw_row_count = len(df)
        duplicate_member_count = int(df['member_id'].duplicated().sum())

        source_intervention = to_binary(df[TREATMENT_COL])
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
        df[TREATMENT_COL] = np.select([resolved_treated, resolved_control], [1.0, 0.0], default=np.nan)
        df['treatment_derivation_status'] = np.select(
            [contradictory_treatment, resolved_treated, resolved_control],
            ['contradictory', 'treated', 'control'],
            default='unresolved',
        )

        outcome_numeric = pd.to_numeric(df[OUTCOME_COL], errors='coerce')
        missing_outcome_before_fill = int(outcome_numeric.isna().sum())
        if (outcome_numeric.dropna() < 0).any():
            raise ValueError(f'{OUTCOME_COL} contains negative values.')
        raw_outcome_count_distribution = outcome_numeric.fillna(0.0).value_counts().sort_index().to_dict()
        positive_counts_binarized = int((outcome_numeric.fillna(0.0) > 0).sum())
        df[OUTCOME_COL] = (outcome_numeric.fillna(0.0) > 0).astype(float)

        death_text = df['date_of_death'].astype('string').str.strip()
        df['death_within_90d_flag'] = (death_text.notna() & death_text.ne('')).astype(float)

        risk_numeric = pd.to_numeric(df['risk_tier'], errors='coerce')
        missing_risk_tier_rows = int(risk_numeric.isna().sum())
        invalid_risk_tier_rows = int((risk_numeric.notna() & ~risk_numeric.isin([1, 2, 3, 4, 5])).sum())
        valid_risk_tier = risk_numeric.where(risk_numeric.isin([1, 2, 3, 4, 5]))
        df['risk_tier'] = valid_risk_tier.astype('Int64').astype('string').fillna('Missing')

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
            canonical_column_name(TREATMENT_COL),
        }

        candidate_predictors = [column for column in candidate_predictors_all if column in df.columns]
        missing_predictors = [column for column in candidate_predictors_all if column not in df.columns]
        exclusion_reasons = {}
        for column in df.columns:
            canonical = canonical_column_name(column)
            if column in candidate_predictors:
                continue
            if canonical in prohibited_canonical or (column.startswith('outcome_') and column != OUTCOME_COL):
                exclusion_reasons[column] = 'post-treatment, alternate outcome, or explicitly prohibited'
            elif canonical in raw_treatment_canonical:
                exclusion_reasons[column] = 'treatment definition/source; metadata only'
            elif column == 'member_id':
                exclusion_reasons[column] = 'identifier retained in outputs; never modeled'
            elif column == 'treatment_derivation_status':
                exclusion_reasons[column] = 'treatment derivation audit metadata'
            else:
                exclusion_reasons[column] = 'not on leakage-safe baseline predictor allowlist'

        meta_columns = [
            'member_id', 'source_intervention_flag', 'optout_flag',
            'treatment_derivation_status', OUTCOME_COL, TREATMENT_COL,
        ]
        model_df = df[[*meta_columns, *candidate_predictors]].copy()
        unresolved_mask = ~model_df['treatment_derivation_status'].isin(['treated', 'control'])
        excluded_treatment_rows = int(unresolved_mask.sum())
        model_df = model_df.loc[~unresolved_mask].copy().reset_index(drop=True)

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

        numeric_predictors = [
            'age', 'op_visits_last_6m', 'ed_visits_last_30d', 'ed_visits_last_6m',
            'admits_last_6m', 'observation_stays_last_6m', 'total_cost_last_6m',
            'rx_count_last_6m', 'current_risk_score', 'percolator_score',
        ]
        for column in present_columns(numeric_predictors, model_df):
            model_df[column] = pd.to_numeric(model_df[column], errors='coerce')

        for column in candidate_predictors:
            if pd.api.types.is_numeric_dtype(model_df[column]):
                model_df[column] = impute_numeric(model_df[column])
            else:
                model_df[column] = impute_categorical(model_df[column])

        predictor_unique_counts = model_df[candidate_predictors].nunique(dropna=True)
        zero_variance_predictors = predictor_unique_counts[predictor_unique_counts <= 1].index.tolist()
        for column in zero_variance_predictors:
            exclusion_reasons[column] = 'zero-variance predictor removed after preprocessing'
        feature_cols = [column for column in candidate_predictors if column not in zero_variance_predictors]
        model_df = model_df[[*meta_columns, *feature_cols]].copy()
        model_df.insert(0, 'analysis_row_id', np.arange(len(model_df), dtype=int))

        # Keep duplicated source member IDs, but use analysis_row_id for unambiguous comparisons.
        assert model_df['analysis_row_id'].is_unique
        assert int(model_df['death_within_90d_flag'].sum()) > 0
        assert set(model_df[TREATMENT_COL].unique()) == {0.0, 1.0}
        assert set(model_df[OUTCOME_COL].unique()) == {0.0, 1.0}

        predictor_inventory_rows = []
        for column in df.columns:
            included = column in feature_cols
            predictor_inventory_rows.append({
                'feature': column,
                'included_in_model': included,
                'reason': 'leakage-safe baseline predictor' if included else exclusion_reasons.get(column, 'not selected'),
                'source_dtype': str(df[column].dtype),
                'unique_values': int(df[column].nunique(dropna=True)),
            })
        predictor_inventory = pd.DataFrame(predictor_inventory_rows)
        save_csv(predictor_inventory, PATHS['predictor_inventory'])

        preprocessing_column_audit = predictor_inventory.rename(columns={'feature': 'column'})
        save_csv(preprocessing_column_audit, PATHS['preprocessing_column_audit'])

        def predictor_type(column):
            series = model_df[column]
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
                return 'Binary indicator'
            if column in numeric_predictors or pd.api.types.is_numeric_dtype(series):
                return 'Continuous/count numeric'
            return 'Multi-level categorical'

        predictor_types = pd.Series({column: predictor_type(column) for column in feature_cols})
        data_review_summary = pd.DataFrame({
            'metric': [
                'Raw rows', 'Raw columns', 'Modeling members/rows', 'Duplicate member_id observations retained',
                'Treated members', 'Control members', 'Treatment rate', 'ED outcome events',
                'Outcome prevalence', 'Treated observed ED rate', 'Control observed ED rate',
                'Missing outcomes filled with zero', 'Death-flag rows retained',
                'Final predictors before one-hot encoding', 'Continuous/count numeric predictors',
                'Binary indicator predictors', 'Multi-level categorical predictors',
            ],
            'current_value': [
                raw_row_count, len(df_raw.columns), len(model_df), duplicate_member_count,
                int((model_df[TREATMENT_COL] == 1).sum()), int((model_df[TREATMENT_COL] == 0).sum()),
                model_df[TREATMENT_COL].mean(), int((model_df[OUTCOME_COL] == 1).sum()),
                model_df[OUTCOME_COL].mean(), model_df.loc[model_df[TREATMENT_COL] == 1, OUTCOME_COL].mean(),
                model_df.loc[model_df[TREATMENT_COL] == 0, OUTCOME_COL].mean(), missing_outcome_before_fill,
                int(model_df['death_within_90d_flag'].sum()), len(feature_cols),
                int((predictor_types == 'Continuous/count numeric').sum()),
                int((predictor_types == 'Binary indicator').sum()),
                int((predictor_types == 'Multi-level categorical').sum()),
            ],
        })
        save_csv(data_review_summary, PATHS['data_review_summary'])

        print('Treatment derivation status:')
        display(df['treatment_derivation_status'].value_counts(dropna=False).rename_axis('status').to_frame('n'))
        print('Final Funds predictors:')
        print(feature_cols)
        display(data_review_summary)
        """
    ),
    markdown(
        """
        ## Analytical Task 3: Causal Forest Diagnostics And Estimation Credibility

        The diagnostics cover stratified train/test composition, event counts, treatment
        overlap, and propensity behavior. The propensity specification exactly matches the
        fixed Funds uplift model (elastic-net logistic, `l1_ratio=0.5`, `C=1.0`).
        """
    ),
    code(
        """
        feature_frame = model_df[feature_cols].copy()
        _, [x_all] = make_design_matrix([feature_frame])

        forbidden_feature_canonical = prohibited_canonical | raw_treatment_canonical
        leaking_features = [
            column for column in feature_cols
            if any(canonical_column_name(column).startswith(item) for item in forbidden_feature_canonical)
        ]
        leaking_matrix_columns = [
            column for column in x_all.columns
            if any(canonical_column_name(column).startswith(item) for item in forbidden_feature_canonical)
        ]
        assert not leaking_features, f'Prohibited predictors found: {leaking_features}'
        assert not leaking_matrix_columns, f'Encoded leakage columns found: {leaking_matrix_columns}'
        assert 'member_id' not in feature_cols and 'analysis_row_id' not in feature_cols

        train_df, test_df = split_train_test(
            model_df,
            train_fraction=TRAIN_FRACTION,
            seed=SEED,
            stratify_columns=[TREATMENT_COL, OUTCOME_COL],
        )
        x_train = x_all.loc[train_df.index].reset_index(drop=True)
        x_test = x_all.loc[test_df.index].reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        y_train = train_df[OUTCOME_COL].astype(float).to_numpy()
        w_train = train_df[TREATMENT_COL].astype(float).to_numpy()
        y_test = test_df[OUTCOME_COL].astype(float).to_numpy()
        w_test = test_df[TREATMENT_COL].astype(float).to_numpy()

        for frame_name, frame in [('train', train_df), ('test', test_df)]:
            assert set(frame[TREATMENT_COL].unique()) == {0.0, 1.0}, f'{frame_name}: missing treatment class'
            assert set(frame[OUTCOME_COL].unique()) == {0.0, 1.0}, f'{frame_name}: missing outcome class'
            for treatment_value in [0.0, 1.0]:
                subgroup = frame.loc[frame[TREATMENT_COL] == treatment_value, OUTCOME_COL]
                assert set(subgroup.unique()) == {0.0, 1.0}, f'{frame_name}: treatment subgroup lacks outcome class'

        data_review_summary = pd.concat([
            data_review_summary,
            pd.DataFrame({
                'metric': ['Model matrix columns after one-hot encoding', 'Train rows', 'Test rows'],
                'current_value': [x_all.shape[1], len(train_df), len(test_df)],
            }),
        ], ignore_index=True)
        save_csv(data_review_summary, PATHS['data_review_summary'])

        split_distribution = (
            pd.concat([train_df.assign(split='train'), test_df.assign(split='test')])
            .groupby(['split', TREATMENT_COL, OUTCOME_COL], dropna=False)
            .size().rename('n').reset_index()
        )
        save_csv(split_distribution, PATHS['split_distribution'])

        event_rows = []
        for split_name, frame in [('Train', train_df), ('Test', test_df)]:
            for group_value, group_label in [(1.0, 'Treated'), (0.0, 'Control')]:
                subset = frame[frame[TREATMENT_COL] == group_value]
                positive = int((subset[OUTCOME_COL] == 1).sum())
                n = int(len(subset))
                event_rows.append({
                    'split': split_name, 'group': group_label, 'n': n,
                    'positive_ed_events': positive, 'negative_ed_events': n - positive,
                    'event_rate': positive / n if n else np.nan,
                })
        event_count_summary = pd.DataFrame(event_rows)
        save_csv(event_count_summary, PATHS['event_count_summary'])
        display(event_count_summary)

        propensity_model = fit_funds_propensity_model(x_train, w_train, seed=SEED)
        train_propensity = clipped_propensity(propensity_model, x_train)
        test_propensity = clipped_propensity(propensity_model, x_test)
        all_propensity = clipped_propensity(propensity_model, x_all)
        propensity_series = pd.Series(test_propensity, dtype=float)
        propensity_summary = pd.DataFrame({
            'metric': [
                'Propensity source', 'Train treatment model AUC', 'Test treatment model AUC',
                'Mean propensity', 'Min propensity', '5th percentile', 'Median propensity',
                '95th percentile', 'Max propensity', 'Members at lower clip (0.05)',
                'Members at upper clip (0.95)',
            ],
            'value': [
                'fixed_funds_uplift_matched_elastic_net', propensity_auc(w_train, train_propensity),
                propensity_auc(w_test, test_propensity), propensity_series.mean(), propensity_series.min(),
                propensity_series.quantile(0.05), propensity_series.median(), propensity_series.quantile(0.95),
                propensity_series.max(), int((propensity_series <= 0.05).sum()),
                int((propensity_series >= 0.95).sum()),
            ],
        })
        save_csv(propensity_summary, PATHS['propensity_summary'])
        display(propensity_summary)

        plt.figure(figsize=(8, 4.5))
        plt.hist(test_propensity[w_test == 1], bins=20, alpha=0.65, label='Treated')
        plt.hist(test_propensity[w_test == 0], bins=20, alpha=0.65, label='Control')
        plt.xlabel('Estimated propensity for intervention')
        plt.ylabel('Members')
        plt.title('Funds Causal Forest Propensity Overlap Check')
        plt.legend()
        save_current_figure(PATHS['propensity_chart'])

        preprocessing_audit_summary = pd.DataFrame([{
            'raw_rows': raw_row_count,
            'raw_columns': len(df_raw.columns),
            'duplicate_member_id_observations_retained': duplicate_member_count,
            'missing_outcome_filled_with_zero': missing_outcome_before_fill,
            'positive_outcome_counts_binarized_to_one': positive_counts_binarized,
            'contradictory_treatment_rows': int((df['treatment_derivation_status'] == 'contradictory').sum()),
            'unresolved_treatment_rows': int((df['treatment_derivation_status'] == 'unresolved').sum()),
            'excluded_treatment_rows': excluded_treatment_rows,
            'modeling_rows': len(model_df),
            'treated_rows': int((model_df[TREATMENT_COL] == 1).sum()),
            'control_rows': int((model_df[TREATMENT_COL] == 0).sum()),
            'outcome_positive_rows': int((model_df[OUTCOME_COL] == 1).sum()),
            'outcome_zero_rows': int((model_df[OUTCOME_COL] == 0).sum()),
            'death_flag_rows_retained': int(model_df['death_within_90d_flag'].sum()),
            'missing_source_risk_tier_rows': missing_risk_tier_rows,
            'out_of_range_source_risk_tier_rows': invalid_risk_tier_rows,
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'retained_predictor_count': len(feature_cols),
            'encoded_feature_count': x_all.shape[1],
            'zero_variance_predictors_removed': '; '.join(zero_variance_predictors),
            'retained_predictors': '; '.join(feature_cols),
            'treatment_distribution': json.dumps(model_df[TREATMENT_COL].value_counts().sort_index().to_dict()),
            'binary_outcome_distribution': json.dumps(model_df[OUTCOME_COL].value_counts().sort_index().to_dict()),
            'raw_count_distribution_before_binary': json.dumps(raw_outcome_count_distribution),
            'leakage_assertions_passed': True,
        }])
        save_csv(preprocessing_audit_summary, PATHS['preprocessing_audit_summary'])
        display(preprocessing_audit_summary)
        """
    ),
    markdown(
        """
        ### Fit Causal Forest With Stability-Aware Hyperparameter Grid

        Tuning occurs only inside the training data. Selection balances held-out HTE
        separation with rank and top-group stability across candidate forests; the final
        30% test set remains untouched until reporting.
        """
    ),
    code(
        """
        def make_causal_forest_treatment_model(seed=SEED):
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, penalty='elasticnet', solver='saga', l1_ratio=0.5,
                    max_iter=2000, tol=1e-3, random_state=seed,
                ),
            )

        def make_causal_forest_model(params, seed=SEED):
            return CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=1,
                ),
                model_t=make_causal_forest_treatment_model(seed),
                discrete_treatment=True,
                n_estimators=params['n_estimators'],
                min_samples_leaf=params['min_samples_leaf'],
                max_depth=params['max_depth'],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                random_state=seed,
                n_jobs=1,
            )

        def score_candidate(candidate_model, x_validation):
            tau = np.asarray(candidate_model.effect(x_validation), dtype=float)
            benefit = -tau
            decile = ntile_desc(pd.Series(benefit), 10).to_numpy()
            top_mask, bottom_mask = decile == 1, decile == 10
            top_count = max(1, int(np.ceil(0.20 * len(benefit))))
            return {
                'tau_hat': tau,
                'benefit_score': benefit,
                'avg_benefit_score': float(np.mean(benefit)),
                'benefit_std': float(np.std(benefit, ddof=1)),
                'top_decile_avg_benefit': float(np.mean(benefit[top_mask])),
                'bottom_decile_avg_benefit': float(np.mean(benefit[bottom_mask])),
                'top_bottom_benefit_gap': float(np.mean(benefit[top_mask]) - np.mean(benefit[bottom_mask])),
                'top_decile_member_index': set(np.where(top_mask)[0]),
                'top_20pct_member_index': set(np.argsort(-benefit)[:top_count]),
            }

        def pairwise_stability(candidate_scores):
            rows = []
            candidate_ids = sorted(candidate_scores)
            for position, candidate_a in enumerate(candidate_ids):
                for candidate_b in candidate_ids[position + 1:]:
                    a, b = candidate_scores[candidate_a], candidate_scores[candidate_b]
                    top_a, top_b = a['top_decile_member_index'], b['top_decile_member_index']
                    top20_a, top20_b = a['top_20pct_member_index'], b['top_20pct_member_index']
                    rows.append({
                        'candidate_a': candidate_a,
                        'candidate_b': candidate_b,
                        'spearman_benefit_corr': pd.Series(a['benefit_score']).corr(pd.Series(b['benefit_score']), method='spearman'),
                        'pearson_benefit_corr': pd.Series(a['benefit_score']).corr(pd.Series(b['benefit_score']), method='pearson'),
                        'top_decile_jaccard': len(top_a & top_b) / len(top_a | top_b),
                        'top_20pct_jaccard': len(top20_a & top20_b) / len(top20_a | top20_b),
                    })
            return pd.DataFrame(rows)

        train_for_tuning = train_df.reset_index(drop=True)
        tuning_train_df, tuning_validation_df = split_train_test(
            train_for_tuning, train_fraction=0.80, seed=SEED,
            stratify_columns=[TREATMENT_COL, OUTCOME_COL],
        )
        x_tuning_train = x_train.loc[tuning_train_df.index].reset_index(drop=True)
        x_tuning_validation = x_train.loc[tuning_validation_df.index].reset_index(drop=True)
        y_tuning_train = tuning_train_df[OUTCOME_COL].astype(float).to_numpy()
        w_tuning_train = tuning_train_df[TREATMENT_COL].astype(float).to_numpy()

        causal_forest_param_grid = [
            {'n_estimators': 400, 'min_samples_leaf': 5, 'max_depth': None},
            {'n_estimators': 800, 'min_samples_leaf': 10, 'max_depth': None},
            {'n_estimators': 800, 'min_samples_leaf': 20, 'max_depth': None},
            {'n_estimators': 800, 'min_samples_leaf': 10, 'max_depth': 8},
        ]

        candidate_rows, candidate_scores = [], {}
        for candidate_id, params in enumerate(causal_forest_param_grid, start=1):
            candidate_model = make_causal_forest_model(params, seed=SEED)
            candidate_model.fit(y_tuning_train, w_tuning_train, X=x_tuning_train)
            score = score_candidate(candidate_model, x_tuning_validation)
            candidate_scores[candidate_id] = score
            candidate_rows.append({
                'candidate_id': candidate_id, **params,
                'validation_avg_benefit_score': score['avg_benefit_score'],
                'validation_benefit_std': score['benefit_std'],
                'validation_top_decile_avg_benefit': score['top_decile_avg_benefit'],
                'validation_bottom_decile_avg_benefit': score['bottom_decile_avg_benefit'],
                'validation_top_bottom_benefit_gap': score['top_bottom_benefit_gap'],
            })
            print(f'Finished candidate {candidate_id}: {params}')

        tuning_summary = pd.DataFrame(candidate_rows)
        stability_pairs = pairwise_stability(candidate_scores)
        stability_long = []
        for _, row in stability_pairs.iterrows():
            for side in ['a', 'b']:
                stability_long.append({
                    'candidate_id': int(row[f'candidate_{side}']),
                    'spearman_benefit_corr': row['spearman_benefit_corr'],
                    'pearson_benefit_corr': row['pearson_benefit_corr'],
                    'top_decile_jaccard': row['top_decile_jaccard'],
                    'top_20pct_jaccard': row['top_20pct_jaccard'],
                })
        stability_summary = pd.DataFrame(stability_long).groupby('candidate_id', as_index=False).agg(
            avg_spearman_vs_other_candidates=('spearman_benefit_corr', 'mean'),
            avg_pearson_vs_other_candidates=('pearson_benefit_corr', 'mean'),
            avg_top_decile_jaccard_vs_other_candidates=('top_decile_jaccard', 'mean'),
            avg_top_20pct_jaccard_vs_other_candidates=('top_20pct_jaccard', 'mean'),
        )
        tuning_summary = tuning_summary.merge(stability_summary, on='candidate_id', how='left')
        tuning_summary['separation_rank'] = tuning_summary['validation_top_bottom_benefit_gap'].rank(ascending=False, method='min')
        tuning_summary['stability_rank'] = tuning_summary['avg_spearman_vs_other_candidates'].rank(ascending=False, method='min')
        tuning_summary['top_decile_stability_rank'] = tuning_summary['avg_top_decile_jaccard_vs_other_candidates'].rank(ascending=False, method='min')
        tuning_summary['selection_score'] = (
            0.50 * tuning_summary['separation_rank']
            + 0.30 * tuning_summary['stability_rank']
            + 0.20 * tuning_summary['top_decile_stability_rank']
        )
        tuning_summary = tuning_summary.sort_values(['selection_score', 'separation_rank']).reset_index(drop=True)
        tuning_summary['selected'] = False
        tuning_summary.loc[0, 'selected'] = True
        save_csv(tuning_summary, PATHS['hyperparameter_tuning_summary'])
        save_csv(stability_pairs, PATHS['hyperparameter_stability_pairs'])
        display(tuning_summary)

        selected_params = {
            'n_estimators': int(tuning_summary.loc[0, 'n_estimators']),
            'min_samples_leaf': int(tuning_summary.loc[0, 'min_samples_leaf']),
            'max_depth': None if pd.isna(tuning_summary.loc[0, 'max_depth']) else int(tuning_summary.loc[0, 'max_depth']),
        }
        cf_model = make_causal_forest_model(selected_params, seed=SEED)
        cf_model.fit(y_train, w_train, X=x_train)
        print(f'Causal forest trained successfully: {selected_params}')
        """
    ),
    markdown("## Analytical Task 4: Treatment Effect Analysis"),
    code(
        """
        tau_test = np.asarray(cf_model.effect(x_test), dtype=float)
        tau_se_test = effect_standard_errors(cf_model, x_test)

        results_test = test_df.copy()
        results_test['tau_hat'] = tau_test
        results_test['tau_se'] = tau_se_test
        results_test['benefit_score'] = -results_test['tau_hat']
        results_test['hte_decile'] = ntile_desc(results_test['benefit_score'], 10).to_numpy()
        results_test['uplift_decile'] = results_test['hte_decile']
        results_test['propensity_score'] = test_propensity
        results_test['tau_ci_lower'] = results_test['tau_hat'] - 1.96 * results_test['tau_se']
        results_test['tau_ci_upper'] = results_test['tau_hat'] + 1.96 * results_test['tau_se']
        results_test['benefit_ci_lower'] = -results_test['tau_ci_upper']
        results_test['benefit_ci_upper'] = -results_test['tau_ci_lower']
        save_csv(results_test, PATHS['test_scored_output'])

        true_benefit_validation_summary = pd.DataFrame([{
            'section': 'synthetic_true_benefit_validation',
            'applicability': 'not_applicable',
            'reason': 'Funds Combined does not provide known member-level counterfactual treatment-effect ground truth.',
            'fabricated_ground_truth': False,
        }])
        save_csv(true_benefit_validation_summary, PATHS['true_benefit_validation_summary'])

        ate_summary = pd.DataFrame({
            'metric': ['avg_tau_hat', 'avg_benefit_score', 'test_members'],
            'value': [results_test['tau_hat'].mean(), results_test['benefit_score'].mean(), len(results_test)],
        })
        save_csv(ate_summary, PATHS['ate_summary'])

        effect_distribution_summary = summarize_distribution(results_test['tau_hat'], 'tau_hat').merge(
            summarize_distribution(results_test['benefit_score'], 'benefit_score'), on='metric', how='outer'
        )
        save_csv(effect_distribution_summary, PATHS['effect_distribution_summary'])

        uncertainty_summary = pd.DataFrame({
            'metric': [
                'Mean tau standard error', 'Median tau standard error',
                'Members with tau CI entirely below zero', 'Members with tau CI crossing zero',
                'Members with tau CI entirely above zero', 'Top HTE decile mean tau standard error',
            ],
            'value': [
                results_test['tau_se'].mean(), results_test['tau_se'].median(),
                int((results_test['tau_ci_upper'] < 0).sum()),
                int(((results_test['tau_ci_lower'] <= 0) & (results_test['tau_ci_upper'] >= 0)).sum()),
                int((results_test['tau_ci_lower'] > 0).sum()),
                results_test.loc[results_test['hte_decile'] == 1, 'tau_se'].mean(),
            ],
        })
        save_csv(uncertainty_summary, PATHS['uncertainty_summary'])
        display(ate_summary)
        display(effect_distribution_summary)
        display(uncertainty_summary)

        plt.figure(figsize=(8, 4.5))
        plt.hist(results_test['benefit_score'], bins=24, alpha=0.85)
        plt.axvline(results_test['benefit_score'].mean(), linestyle='--', color='black')
        plt.xlabel('Benefit score (-tau_hat)')
        plt.ylabel('Members')
        plt.title('Funds Causal Forest Estimated Benefit Distribution')
        save_current_figure(PATHS['effect_distribution_chart'])
        """
    ),
    markdown("## Analytical Task 5: HTE Decile And High-Value Subgroup Analysis"),
    code(
        """
        agg_map = {
            'n': ('hte_decile', 'size'),
            'avg_tau_hat': ('tau_hat', 'mean'),
            'avg_benefit_score': ('benefit_score', 'mean'),
            'avg_tau_se': ('tau_se', 'mean'),
            'observed_ed_rate': (OUTCOME_COL, 'mean'),
            'treatment_pct': (TREATMENT_COL, 'mean'),
            'avg_propensity_score': ('propensity_score', 'mean'),
        }
        if 'current_risk_score' in results_test.columns:
            agg_map['avg_current_risk_score'] = ('current_risk_score', 'mean')
        decile_summary = results_test.groupby('hte_decile', as_index=False).agg(**agg_map).sort_values('hte_decile')
        decile_summary['uplift_decile'] = decile_summary['hte_decile']
        save_csv(decile_summary, PATHS['decile_summary'])
        display(decile_summary)

        observed_gap_rows = []
        for decile, group in results_test.groupby('hte_decile'):
            treated = group[group[TREATMENT_COL] == 1]
            control = group[group[TREATMENT_COL] == 0]
            treated_rate = treated[OUTCOME_COL].mean() if len(treated) else np.nan
            control_rate = control[OUTCOME_COL].mean() if len(control) else np.nan
            observed_gap_rows.append({
                'hte_decile': int(decile), 'n': len(group), 'treated_n': len(treated), 'control_n': len(control),
                'avg_predicted_benefit': group['benefit_score'].mean(),
                'treated_observed_ed_rate': treated_rate, 'control_observed_ed_rate': control_rate,
                'observed_control_minus_treated_gap': control_rate - treated_rate,
            })
        observed_gap_summary = pd.DataFrame(observed_gap_rows).sort_values('hte_decile')
        save_csv(observed_gap_summary, PATHS['observed_gap_summary'])

        plt.figure(figsize=(8, 4.5))
        plt.bar(decile_summary['hte_decile'].astype(str), decile_summary['avg_benefit_score'])
        plt.xlabel('HTE decile (1 = highest estimated benefit)')
        plt.ylabel('Average benefit score')
        plt.title('Funds Causal Forest Average Estimated Benefit By HTE Decile')
        save_current_figure(PATHS['benefit_decile_chart'])

        plt.figure(figsize=(8, 4.5))
        plt.bar(decile_summary['hte_decile'].astype(str), decile_summary['avg_tau_hat'])
        plt.axhline(0, linewidth=1, color='black')
        plt.xlabel('HTE decile (1 = highest estimated benefit)')
        plt.ylabel('Average tau_hat')
        plt.title('Funds Causal Forest Average Treatment Effect By HTE Decile')
        save_current_figure(PATHS['tau_decile_chart'])

        plt.figure(figsize=(8, 4.5))
        plt.plot(observed_gap_summary['hte_decile'], observed_gap_summary['avg_predicted_benefit'], marker='o', label='Predicted benefit')
        plt.plot(observed_gap_summary['hte_decile'], observed_gap_summary['observed_control_minus_treated_gap'], marker='o', label='Observed control − treated gap')
        plt.axhline(0, linewidth=1, color='black')
        plt.xlabel('HTE decile (1 = highest estimated benefit)')
        plt.ylabel('ED-rate difference')
        plt.title('Funds Causal Forest Predicted Benefit And Observed Gap')
        plt.legend()
        save_current_figure(PATHS['observed_gap_chart'])

        risk_tier_order = ['1', '2', '3', '4', '5']
        benefit_group_order = ['High benefit', 'Medium benefit', 'Low benefit']
        risk_frame = results_test.copy()
        benefit_rank = risk_frame['benefit_score'].rank(method='first', ascending=False)
        risk_frame['benefit_group'] = pd.qcut(benefit_rank, q=3, labels=benefit_group_order)
        risk_frame = risk_frame[risk_frame['risk_tier'].isin(risk_tier_order)].copy()
        risk_frame['risk_tier'] = pd.Categorical(risk_frame['risk_tier'], categories=risk_tier_order, ordered=True)
        risk_frame['benefit_group'] = pd.Categorical(risk_frame['benefit_group'], categories=benefit_group_order, ordered=True)
        risk_summary = risk_frame.groupby(['risk_tier', 'benefit_group'], observed=False).size().reset_index(name='members')
        tier_totals = risk_frame.groupby('risk_tier', observed=False).size().rename('risk_tier_members').reset_index()
        risk_summary = risk_summary.merge(tier_totals, on='risk_tier', how='left')
        risk_summary['pct_within_risk_tier'] = np.where(
            risk_summary['risk_tier_members'] > 0,
            risk_summary['members'] / risk_summary['risk_tier_members'], np.nan,
        )
        risk_summary['risk_tier_source'] = 'Original Funds Combined assignment; not derived from current_risk_score'
        save_csv(risk_summary, PATHS['risk_tier_benefit_group_summary'])

        risk_pivot = risk_summary.pivot(index='risk_tier', columns='benefit_group', values='pct_within_risk_tier').fillna(0)
        risk_pivot = risk_pivot.reindex(risk_tier_order)
        risk_pivot.plot(kind='bar', stacked=True, figsize=(9, 5.5), color=['#1f77b4', '#70ad47', '#a6a6a6'])
        plt.xlabel('Original Funds risk tier (1 = lowest, 5 = highest)')
        plt.ylabel('Percent within risk tier')
        plt.title('Funds Causal Forest Benefit Groups Within Original Risk Tiers')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
        save_current_figure(PATHS['risk_tier_benefit_group_chart'])
        """
    ),
    code(
        """
        ranked = results_test.sort_values('benefit_score', ascending=False)
        example_rows = [
            ('Highest benefit', ranked.iloc[0], 'Strongest outreach candidate by estimated ED risk reduction.'),
            ('Lowest benefit', ranked.iloc[-1], 'Lowest priority by causal-forest benefit score.'),
        ]
        high_risk = ranked['current_risk_score'].quantile(0.75)
        low_benefit = ranked['benefit_score'].quantile(0.25)
        subset = ranked[(ranked['current_risk_score'] >= high_risk) & (ranked['benefit_score'] <= low_benefit)]
        if not subset.empty:
            example_rows.append(('High risk, low benefit', subset.iloc[0], 'High baseline risk but limited estimated impactability.'))
        low_risk = ranked['current_risk_score'].quantile(0.50)
        high_benefit = ranked['benefit_score'].quantile(0.75)
        subset = ranked[(ranked['current_risk_score'] <= low_risk) & (ranked['benefit_score'] >= high_benefit)]
        if not subset.empty:
            example_rows.append(('Lower risk, high benefit', subset.iloc[0], 'May be missed by risk-only targeting but appears impactable.'))

        top_benefit_examples = pd.DataFrame([{
            'member_profile': label, 'analysis_row_id': int(row['analysis_row_id']),
            'member_id': row['member_id'], 'actual_outcome': row[OUTCOME_COL],
            'treatment_flag': row[TREATMENT_COL], 'current_risk_score': row['current_risk_score'],
            'risk_tier': row['risk_tier'], 'tau_hat': row['tau_hat'], 'tau_se': row['tau_se'],
            'benefit_score': row['benefit_score'], 'hte_decile': row['hte_decile'],
            'outreach_interpretation': interpretation,
        } for label, row, interpretation in example_rows])
        save_csv(top_benefit_examples, PATHS['top_benefit_examples'])
        display(top_benefit_examples)
        """
    ),
    markdown(
        """
        ### Framework Consistency Check

        Comparisons use Funds uplift outputs only. Because the source `member_id` is not
        unique, full-file T-learner outputs are aligned by the deterministic source-row
        order and test-only X-learner outputs are aligned by the reproduced stratified
        test order after validating member, treatment, and outcome sequences. This avoids
        invalid many-to-many member-ID joins.
        """
    ),
    code(
        """
        tau_full = np.asarray(cf_model.effect(x_all), dtype=float)
        tau_se_full = effect_standard_errors(cf_model, x_all)
        scored_full = model_df.copy()
        scored_full['tau_hat'] = tau_full
        scored_full['tau_se'] = tau_se_full
        scored_full['benefit_score'] = -scored_full['tau_hat']
        scored_full['hte_decile'] = ntile_desc(scored_full['benefit_score'], 10).to_numpy()
        scored_full['uplift_decile'] = scored_full['hte_decile']
        scored_full['propensity_score'] = all_propensity
        save_csv(scored_full, PATHS['scored_output'])

        def aligned_benchmark(path, scope):
            if not path.exists():
                print(f'Skipping missing benchmark: {path}')
                return None, 'missing'
            benchmark = pd.read_csv(path)
            required = {'member_id', TREATMENT_COL, OUTCOME_COL, 'benefit_score'}
            if not required.issubset(benchmark.columns):
                print(f'Skipping benchmark missing columns {required - set(benchmark.columns)}: {path}')
                return None, 'missing_required_columns'
            if scope == 'full':
                if len(benchmark) != len(model_df):
                    return None, 'full_row_count_mismatch'
                reference = model_df
                benchmark['analysis_row_id'] = np.arange(len(benchmark), dtype=int)
                basis = 'validated_full_source_row_order'
            else:
                if len(benchmark) != len(results_test):
                    return None, 'test_row_count_mismatch'
                reference = results_test
                benchmark['analysis_row_id'] = results_test['analysis_row_id'].to_numpy()
                basis = 'validated_reproduced_test_row_order'
            member_match = benchmark['member_id'].astype(str).to_numpy() == reference['member_id'].astype(str).to_numpy()
            treatment_match = np.isclose(benchmark[TREATMENT_COL].astype(float), reference[TREATMENT_COL].astype(float))
            outcome_match = np.isclose(benchmark[OUTCOME_COL].astype(float), reference[OUTCOME_COL].astype(float))
            if not (member_match.all() and treatment_match.all() and outcome_match.all()):
                return None, 'row_sequence_validation_failed'
            if scope == 'full':
                benchmark = benchmark[benchmark['analysis_row_id'].isin(results_test['analysis_row_id'])]
            return benchmark[['analysis_row_id', 'benefit_score']], basis

        uplift_root = PROJECT_ROOT / 'Outputs' / 'Uplift_Funds' / 'Python'
        benchmark_specs = [
            ('XGBoost', 'T-Learner', 'Primary selected Funds uplift benchmark', uplift_root / 'T-Learner' / 'XGBoost' / 'uplift_scored_output.csv', 'full'),
            ('XGBoost', 'X-Learner', 'Primary selected Funds uplift benchmark', uplift_root / 'X-Learner' / 'XGBoost' / 'xlearner_scored_test_output.csv', 'test'),
            ('GLMNet', 'T-Learner', 'Fixed-specification sensitivity benchmark', uplift_root / 'T-Learner' / 'GLMNet' / 'uplift_scored_output.csv', 'full'),
            ('GLMNet', 'X-Learner', 'Fixed-specification sensitivity benchmark', uplift_root / 'X-Learner' / 'GLMNet' / 'xlearner_scored_test_output.csv', 'test'),
        ]
        comparison_rows = []
        for model_label, framework, role, path, scope in benchmark_specs:
            benchmark, basis = aligned_benchmark(path, scope)
            if benchmark is None:
                comparison_rows.append({
                    'comparison': f'Causal forest vs {model_label} {framework}', 'benchmark_role': role,
                    'benchmark_framework': framework, 'benchmark_model': model_label,
                    'comparison_basis': basis, 'n_compared': 0,
                    'pearson_corr': np.nan, 'spearman_corr': np.nan,
                    'top_decile_overlap': np.nan, 'top_20pct_overlap': np.nan,
                })
                continue
            merged = results_test[['analysis_row_id', 'benefit_score']].merge(
                benchmark.rename(columns={'benefit_score': 'benchmark_benefit_score'}),
                on='analysis_row_id', how='inner', validate='one_to_one',
            )
            comparison_rows.append({
                'comparison': f'Causal forest vs {model_label} {framework}', 'benchmark_role': role,
                'benchmark_framework': framework, 'benchmark_model': model_label,
                'comparison_basis': basis, 'n_compared': len(merged),
                'pearson_corr': safe_corr(merged['benefit_score'], merged['benchmark_benefit_score'], 'pearson'),
                'spearman_corr': safe_corr(merged['benefit_score'], merged['benchmark_benefit_score'], 'spearman'),
                'top_decile_overlap': top_overlap(merged['benefit_score'], merged['benchmark_benefit_score'], 0.10),
                'top_20pct_overlap': top_overlap(merged['benefit_score'], merged['benchmark_benefit_score'], 0.20),
            })
        consistency_summary = pd.DataFrame(comparison_rows)
        save_csv(consistency_summary, PATHS['consistency_summary'])
        display(consistency_summary)
        """
    ),
    markdown("## Analytical Task 6: Variable Importance And Explainability"),
    code(
        """
        importances = getattr(cf_model, 'feature_importances_', np.full(x_train.shape[1], np.nan))
        importance_df = pd.DataFrame({
            'feature': x_train.columns, 'importance': np.asarray(importances, dtype=float),
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        importance_df.insert(0, 'rank', np.arange(1, len(importance_df) + 1))
        save_csv(importance_df, PATHS['variable_importance'])
        display(importance_df.head(20))

        plot_df = importance_df.head(15).sort_values('importance')
        plt.figure(figsize=(9, 6))
        plt.barh(plot_df['feature'], plot_df['importance'])
        plt.xlabel('Causal forest importance')
        plt.ylabel('Encoded feature')
        plt.title('Funds: Top Causal Forest HTE Split Features')
        save_current_figure(PATHS['variable_importance_chart'])

        profile_features = [
            'current_risk_score', 'percolator_score', 'ed_visits_last_30d', 'ed_visits_last_6m',
            'admits_last_6m', 'total_cost_last_6m', 'op_visits_last_6m', 'rx_count_last_6m',
            'bh_flag', 'food_insecurity_flag', 'housing_instability_flag',
            'transportation_barrier_flag', 'financial_strain_flag', 'dual_elg',
            'death_within_90d_flag',
        ]
        profile_features = [feature for feature in profile_features if feature in results_test.columns]
        top_decile_mask = results_test['hte_decile'] == 1
        top_decile_profile = pd.DataFrame([{
            'feature': feature,
            'top_hte_decile_mean_or_rate': pd.to_numeric(results_test.loc[top_decile_mask, feature], errors='coerce').mean(),
            'other_deciles_mean_or_rate': pd.to_numeric(results_test.loc[~top_decile_mask, feature], errors='coerce').mean(),
        } for feature in profile_features])
        top_decile_profile['difference'] = top_decile_profile['top_hte_decile_mean_or_rate'] - top_decile_profile['other_deciles_mean_or_rate']
        top_decile_profile = top_decile_profile.sort_values('difference', key=lambda values: values.abs(), ascending=False)
        save_csv(top_decile_profile, PATHS['top_decile_profile'])
        display(top_decile_profile)
        """
    ),
    markdown(
        """
        ### SHAP Benefit-Score Contributions

        Permutation SHAP explains the final `benefit_score = -tau_hat` function. A
        deterministic, capped held-out sample is used because Funds has hundreds of
        encoded features; the sample size is recorded in the output and does not affect
        model fitting or member scoring.
        """
    ),
    code(
        """
        import shap

        shap_rng = np.random.default_rng(SEED)
        shap_count = min(SHAP_MAX_EXPLAIN_ROWS, len(x_test))
        shap_positions = np.sort(shap_rng.choice(len(x_test), size=shap_count, replace=False))
        x_shap = x_test.iloc[shap_positions].reset_index(drop=True)
        shap_members = results_test.iloc[shap_positions][['analysis_row_id', 'member_id']].reset_index(drop=True)
        background_count = min(50, len(x_train))
        background_positions = np.sort(shap_rng.choice(len(x_train), size=background_count, replace=False))
        background_sample = x_train.iloc[background_positions].reset_index(drop=True)

        def cf_benefit_predict(x_values):
            x_frame = pd.DataFrame(x_values, columns=x_test.columns) if not isinstance(x_values, pd.DataFrame) else x_values
            return -np.asarray(cf_model.effect(x_frame), dtype=float)

        masker = shap.maskers.Independent(background_sample)
        explainer = shap.Explainer(cf_benefit_predict, masker, algorithm='permutation')
        explanation = explainer(x_shap, max_evals=2 * len(x_test.columns) + 1)
        shap_array = np.asarray(explanation.values, dtype=float)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]
        shap_df = pd.DataFrame(shap_array, columns=x_test.columns)

        positive_only = shap_df.where(shap_df > 0, 0.0)
        negative_only = shap_df.where(shap_df < 0, 0.0)
        cf_shap_importance = pd.DataFrame({
            'feature': x_test.columns,
            'mean_abs_benefit_shap': shap_df.abs().mean(axis=0).to_numpy(),
            'mean_signed_benefit_shap': shap_df.mean(axis=0).to_numpy(),
            'mean_positive_benefit_shap': positive_only.mean(axis=0).to_numpy(),
            'mean_negative_benefit_shap': negative_only.mean(axis=0).to_numpy(),
            'pct_positive_benefit_shap': shap_df.gt(0).mean(axis=0).to_numpy(),
            'pct_negative_benefit_shap': shap_df.lt(0).mean(axis=0).to_numpy(),
            'explained_test_rows': shap_count,
            'background_rows': background_count,
        }).sort_values('mean_abs_benefit_shap', ascending=False).reset_index(drop=True)
        save_csv(cf_shap_importance, PATHS['shap_importance'])
        member_shap = pd.concat([shap_members, shap_df], axis=1)
        save_csv(member_shap, PATHS['shap_member_values'])

        top_shap = cf_shap_importance.head(15).sort_values('mean_abs_benefit_shap')
        plt.figure(figsize=(9, 6))
        plt.barh(top_shap['feature'], top_shap['mean_abs_benefit_shap'])
        plt.title('Funds Causal Forest: Global SHAP Drivers Of Benefit Score')
        plt.xlabel('Mean absolute SHAP contribution to benefit score')
        plt.ylabel('Encoded feature')
        save_current_figure(PATHS['shap_chart'])
        display(cf_shap_importance.head(15))
        """
    ),
    markdown("## Analytical Task 7: Business Value Assessment"),
    code(
        """
        targeting_summary = decile_summary[['hte_decile', 'n', 'avg_benefit_score', 'observed_ed_rate', 'treatment_pct']].copy()
        if 'avg_current_risk_score' in decile_summary.columns:
            targeting_summary['avg_current_risk_score'] = decile_summary['avg_current_risk_score']
        targeting_summary['expected_ed_visits_avoided'] = targeting_summary['avg_benefit_score'] * targeting_summary['n']
        targeting_summary['gross_savings'] = targeting_summary['expected_ed_visits_avoided'] * ED_VISIT_COST
        targeting_summary['intervention_cost'] = targeting_summary['n'] * INTERVENTION_COST
        targeting_summary['net_savings'] = targeting_summary['gross_savings'] - targeting_summary['intervention_cost']
        targeting_summary['roi'] = targeting_summary['net_savings'] / targeting_summary['intervention_cost']
        targeting_summary['cumulative_members'] = targeting_summary['n'].cumsum()
        targeting_summary['cumulative_expected_ed_reductions'] = targeting_summary['expected_ed_visits_avoided'].cumsum()
        save_csv(targeting_summary, PATHS['targeting_summary'])

        n_total = len(results_test)
        decile_size = max(1, n_total // 10)
        cumulative_rows = []
        for approach, sort_column in [('Causal forest benefit score', 'benefit_score'), ('Current risk score', 'current_risk_score')]:
            ordered = results_test.sort_values(sort_column, ascending=False).reset_index(drop=True)
            for through_decile in range(1, 11):
                top_n = n_total if through_decile == 10 else min(n_total, through_decile * decile_size)
                selected = ordered.head(top_n)
                avoided = selected['benefit_score'].sum()
                cumulative_rows.append({
                    'targeting_approach': approach, 'through_decile': through_decile,
                    'population_fraction_targeted': top_n / n_total, 'n': top_n,
                    'cumulative_estimated_ed_visits_avoided': avoided,
                    'cumulative_gross_savings': avoided * ED_VISIT_COST,
                })
        cumulative_targeting = pd.DataFrame(cumulative_rows)
        save_csv(cumulative_targeting, PATHS['cumulative_targeting'])

        marginal_frames = []
        for approach, group in cumulative_targeting.groupby('targeting_approach'):
            group = group.sort_values('through_decile').copy()
            group['marginal_gross_savings'] = group['cumulative_gross_savings'].diff().fillna(group['cumulative_gross_savings'])
            marginal_frames.append(group[['targeting_approach', 'through_decile', 'marginal_gross_savings', 'cumulative_gross_savings']])
        marginal_targeting = pd.concat(marginal_frames, ignore_index=True)
        save_csv(marginal_targeting, PATHS['marginal_targeting'])

        benefit_marginal = marginal_targeting[marginal_targeting['targeting_approach'] == 'Causal forest benefit score'].set_index('through_decile')['marginal_gross_savings']
        risk_marginal = marginal_targeting[marginal_targeting['targeting_approach'] == 'Current risk score'].set_index('through_decile')['marginal_gross_savings']
        marginal_advantage = pd.DataFrame({
            'through_decile': range(1, 11),
            'causal_forest_marginal_gross_savings': [benefit_marginal.get(i, np.nan) for i in range(1, 11)],
            'current_risk_marginal_gross_savings': [risk_marginal.get(i, np.nan) for i in range(1, 11)],
        })
        marginal_advantage['marginal_advantage_vs_current_risk'] = (
            marginal_advantage['causal_forest_marginal_gross_savings']
            - marginal_advantage['current_risk_marginal_gross_savings']
        )
        save_csv(marginal_advantage, PATHS['marginal_advantage'])

        plt.figure(figsize=(8.5, 5.2))
        for approach, group in cumulative_targeting[cumulative_targeting['population_fraction_targeted'] <= 0.51].groupby('targeting_approach'):
            plt.plot(group['population_fraction_targeted'], group['cumulative_gross_savings'], marker='o', label=approach)
        plt.xlabel('Population fraction targeted')
        plt.ylabel('Cumulative gross savings ($)')
        plt.title('Funds Causal Forest: Cumulative Gross Savings By Targeting Approach')
        plt.legend()
        plt.grid(alpha=0.3)
        save_current_figure(PATHS['cumulative_targeting_chart'])

        plt.figure(figsize=(8.5, 5.2))
        plt.bar(marginal_advantage['through_decile'].astype(str), marginal_advantage['marginal_advantage_vs_current_risk'])
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel('Targeting decile')
        plt.ylabel('Marginal gross savings advantage ($)')
        plt.title('Funds Causal Forest: Marginal Advantage Versus Current-Risk Targeting')
        save_current_figure(PATHS['marginal_advantage_chart'])
        display(targeting_summary)
        """
    ),
    markdown(
        """
        ## Analytical Task 8: Client Perspective

        The Funds causal forest should be treated as a challenger and subgroup-discovery
        model until ranking stability, overlap, uncertainty, cross-framework consistency,
        and prospective operational performance are judged acceptable. Predicted benefit
        is distinct from baseline risk and should not be interpreted as a guaranteed
        member-level causal effect.

        ## Recommendation

        Use the generated diagnostics to decide whether causal forest adds actionable
        ranking information beyond the stronger Funds XGBoost uplift models. Preserve the
        original Funds risk tiers for operational context, but use benefit estimates—not
        risk alone—when evaluating impactability.

        ## Presentation Summary

        Present the business question, Funds preprocessing, overlap and event support,
        tuning stability, held-out benefit distribution, HTE deciles, uncertainty,
        cross-framework agreement, explainability, and targeting economics in that order.
        """
    ),
    markdown(
        """
        ## Reproducibility

        Primary notebook: `Code/PRISM_Causal_Forest_Modeling_Funds_Workflow.ipynb`<br>
        Input: `DataSets/Funds_combined.csv`<br>
        Output folder: `Outputs/Causal-Forests_Funds/Python`<br>
        Seed: `123`<br>
        Treatment and outcome derivations: matched to `Uplift Model Code_Funds_Combined.ipynb`<br>
        Report: `PRISM_Causal_Forest_Modeling_Funds_README.md`

        Synthetic true-benefit validation is explicitly not applicable because Funds
        Combined contains no known member-level counterfactual ground truth.
        """
    ),
    code(
        """
        expected_outputs = list(PATHS.values())
        output_index = pd.DataFrame({
            'output_name': list(PATHS.keys()),
            'output_path': [str(path) for path in expected_outputs],
            'exists': [path.exists() for path in expected_outputs],
        })
        # The index itself is written after checking the other expected artifacts.
        output_index.loc[output_index['output_name'] == 'output_index', 'exists'] = True
        save_csv(output_index, PATHS['output_index'])

        assert not leaking_features
        assert not leaking_matrix_columns
        assert missing_outcome_before_fill == int(preprocessing_audit_summary.loc[0, 'missing_outcome_filled_with_zero'])
        assert int(model_df['death_within_90d_flag'].sum()) == int(preprocessing_audit_summary.loc[0, 'death_flag_rows_retained'])
        assert 'date_of_death' not in feature_cols
        assert not any('date_of_death' in feature for feature in importance_df['feature'].astype(str))
        assert output_index['exists'].all(), output_index.loc[~output_index['exists']]

        print(f'All {len(output_index)} expected Funds causal-forest artifacts exist.')
        print('Leakage, outcome-fill, death-retention, and output-completeness checks passed.')
        display(output_index)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.13 (econml)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Built {TARGET} with {len(cells)} cells.")
