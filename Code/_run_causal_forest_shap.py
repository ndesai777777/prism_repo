"""
Standalone script to compute SHAP benefit-score contributions for the causal forest model.
Reproduces the same model fit as PRISM_Causal_Forest_Modeling_Workflow.ipynb, then computes
permutation SHAP on the benefit_score = -tau_hat function over the test set.
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from _prism_model_utils import (
    add_date_features, clean_names_simple, ensure_output_folder,
    make_design_matrix, ntile_desc, prepare_model_frame,
    read_prism_excel, require_columns, split_train_test, to_binary,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
import shap

warnings.filterwarnings('ignore')

SEED = 123
TRAIN_FRACTION = 0.70
OUTCOME_COL = 'outcome_ed_90d'
TREATMENT_COL = 'intervention_flag'
OUTPUT_DIR = ensure_output_folder(PROJECT_ROOT / 'Outputs' / 'Causal-Forests' / 'Python')

np.random.seed(SEED)

# --- Data prep (same as notebook) ---
print("Loading data...")
df = read_prism_excel()
df.columns = clean_names_simple(df.columns)
df = df.copy()
require_columns(df, [OUTCOME_COL, TREATMENT_COL])
df[OUTCOME_COL] = to_binary(df[OUTCOME_COL])
df[TREATMENT_COL] = to_binary(df[TREATMENT_COL])
df = add_date_features(df, include_duration=False)

PREDICTOR_CATEGORIES = {
    'demographics': ['client_contract', 'service_region', 'program', 'case_manager_name', 'age', 'gender', 'dual_eligible', 'county', 'plan_type', 'language', 'living_alone_flag'],
    'clinical_conditions': ['diabetes_flag', 'chf_flag', 'copd_flag', 'asthma_flag', 'depression_flag', 'anxiety_flag', 'substance_use_flag', 'ckd_flag', 'behavioral_health_risk_flag'],
    'sdoh': ['food_insecurity_flag', 'housing_instability_flag', 'transportation_barrier_flag', 'utilities_insecurity_flag'],
    'utilization': ['pcp_visits_last_6m', 'specialist_visits_last_6m', 'ed_visits_last_30d', 'ed_visits_last_6m', 'admits_last_6m', 'observation_stays_last_6m'],
    'pharmacy': ['total_cost_last_6m', 'rx_count_last_6m', 'med_adherence_pdc', 'high_cost_drug_flag', 'opioid_flag', 'polypharmacy_flag'],
    'risk_scores': ['percolator_utilization_score', 'percolator_clinical_score', 'percolator_sdoh_score', 'current_risk_score', 'risk_tier'],
}
PREDICTOR_VARS = [f for fs in PREDICTOR_CATEGORIES.values() for f in fs]
NUMERIC_VARS = ['age', 'pcp_visits_last_6m', 'specialist_visits_last_6m', 'ed_visits_last_30d', 'ed_visits_last_6m', 'admits_last_6m', 'observation_stays_last_6m', 'total_cost_last_6m', 'rx_count_last_6m', 'med_adherence_pdc', 'percolator_utilization_score', 'percolator_clinical_score', 'percolator_sdoh_score', 'current_risk_score', 'intervention_start_month', 'intervention_start_wday', 'days_to_intervention_start']
BINARY_EXTRA = ['dual_eligible', 'living_alone_flag']
present_predictors = [f for f in PREDICTOR_VARS if f in df.columns]

model_df = prepare_model_frame(df, present_predictors, NUMERIC_VARS, BINARY_EXTRA)
model_df.insert(0, 'member_id', np.arange(len(model_df), dtype=int))
feature_frame = model_df.drop(columns=['member_id', OUTCOME_COL, TREATMENT_COL])
_, [x_all] = make_design_matrix([feature_frame])

train_df, test_df = split_train_test(model_df, train_fraction=TRAIN_FRACTION, seed=SEED, stratify_columns=[TREATMENT_COL, OUTCOME_COL])
x_train = x_all.loc[train_df.index].reset_index(drop=True)
x_test = x_all.loc[test_df.index].reset_index(drop=True)
y_train = train_df[OUTCOME_COL].astype(float).to_numpy()
w_train = train_df[TREATMENT_COL].astype(float).to_numpy()

print(f"Train: {len(x_train)} rows, Test: {len(x_test)} rows, Features: {x_train.shape[1]}")

# --- Fit causal forest (same selected params as notebook) ---
print("Fitting causal forest model...")
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1),
    model_t=make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 30),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            penalty='elasticnet', solver='saga', l1_ratios=[0.5],
            scoring='roc_auc', max_iter=10000, random_state=SEED, refit=True,
        )
    ),
    discrete_treatment=True,
    n_estimators=800,
    min_samples_leaf=10,
    max_depth=None,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    random_state=SEED,
)
cf_model.fit(y_train, w_train, X=x_train)
print("Causal forest fitted.")

# --- Compute SHAP ---
print("Computing SHAP values (permutation)...")

def cf_benefit_predict(x_values):
    x_df = pd.DataFrame(x_values, columns=x_test.columns) if not isinstance(x_values, pd.DataFrame) else x_values
    tau = np.asarray(cf_model.effect(x_df), dtype=float)
    return -tau

background_sample = shap.sample(x_train, min(100, len(x_train)), random_state=SEED)
masker = shap.maskers.Independent(background_sample)
explainer = shap.Explainer(cf_benefit_predict, masker, algorithm='permutation')
explanation = explainer(x_test, max_evals=2 * len(x_test.columns) + 1)

shap_array = np.asarray(explanation.values, dtype=float)
if shap_array.ndim == 3:
    shap_array = shap_array[:, :, 0]
shap_df = pd.DataFrame(shap_array, columns=x_test.columns)

# Global importance
feature_columns = list(x_test.columns)
positive_only = shap_df.where(shap_df > 0, 0.0)
negative_only = shap_df.where(shap_df < 0, 0.0)

cf_shap_importance = pd.DataFrame({
    'feature': feature_columns,
    'mean_abs_benefit_shap': shap_df.abs().mean(axis=0).to_numpy(),
    'mean_signed_benefit_shap': shap_df.mean(axis=0).to_numpy(),
    'mean_positive_benefit_shap': positive_only.mean(axis=0).to_numpy(),
    'mean_negative_benefit_shap': negative_only.mean(axis=0).to_numpy(),
    'pct_positive_benefit_shap': shap_df.gt(0).mean(axis=0).to_numpy(),
    'pct_negative_benefit_shap': shap_df.lt(0).mean(axis=0).to_numpy(),
}).sort_values('mean_abs_benefit_shap', ascending=False).reset_index(drop=True)

# Save CSVs
shap_importance_path = OUTPUT_DIR / 'causal_forest_global_benefit_shap_importance.csv'
shap_member_path = OUTPUT_DIR / 'causal_forest_member_benefit_shap_values.csv'
shap_chart_path = OUTPUT_DIR / 'dashboard_causal_forest_global_benefit_shap.png'

cf_shap_importance.to_csv(shap_importance_path, index=False)
shap_df.to_csv(shap_member_path, index=True, index_label='row_index')
print(f"Saved: {shap_importance_path}")
print(f"Saved: {shap_member_path}")

# Bar chart
top_shap = cf_shap_importance.head(10).sort_values('mean_abs_benefit_shap', ascending=True)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_shap['feature'], top_shap['mean_abs_benefit_shap'])
ax.set_title('Causal Forest: Global SHAP Drivers of Benefit Score')
ax.set_xlabel('Mean absolute SHAP contribution to benefit score')
ax.set_ylabel('Feature')
fig.tight_layout()
fig.savefig(shap_chart_path, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {shap_chart_path}")

# Print top 10
print("\nTop 10 features by mean absolute SHAP contribution:")
print(cf_shap_importance.head(10)[['feature', 'mean_abs_benefit_shap', 'mean_signed_benefit_shap']].to_string(index=False))

# Signed direction table
positive_drivers = cf_shap_importance[cf_shap_importance['mean_signed_benefit_shap'] > 0].head(5).copy()
positive_drivers['direction'] = 'Increase predicted benefit'
negative_drivers = cf_shap_importance[cf_shap_importance['mean_signed_benefit_shap'] < 0].sort_values('mean_signed_benefit_shap').head(5).copy()
negative_drivers['direction'] = 'Decrease predicted benefit'
signed_table = pd.concat([positive_drivers, negative_drivers], ignore_index=True)[['direction', 'feature', 'mean_signed_benefit_shap']]
print("\nSigned SHAP direction table:")
print(signed_table.to_string(index=False))
print("\nDone.")
