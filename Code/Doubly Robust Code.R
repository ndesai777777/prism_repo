# ============================================================
# DOUBLY ROBUST OFF-POLICY EVALUATION
# Outcome: outcome_ed_90d
# Treatment: intervention_flag
# Lower policy_value = lower expected ED rate
# ============================================================

required_packages <- c("readxl", "readr", "dplyr", "lubridate", "xgboost")

installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}

library(readxl)
library(readr)
library(dplyr)
library(lubridate)
library(xgboost)

# ============================================================
# FILE PATHS
# ============================================================

file_path <- "D:/Users/ben.novinger/OneDrive - Acentra/Analytics Opportunities Workgroup/AI Innovation Challenge/expanded_case_management_dataset_500_rows.xlsx"

output_path <- "D:/Users/ben.novinger/OneDrive - Acentra/Analytics Opportunities Workgroup/AI Innovation Challenge/doubly_robust_policy_evaluation.csv"

scored_output_path <- "D:/Users/ben.novinger/OneDrive - Acentra/Analytics Opportunities Workgroup/AI Innovation Challenge/doubly_robust_scored_members.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

clean_names_simple <- function(x) {
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x <- gsub("_+", "_", x)
  x
}

to_binary <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  
  x_chr <- trimws(tolower(as.character(x)))
  
  out <- ifelse(
    x_chr %in% c("1", "y", "yes", "true", "t"), 1,
    ifelse(
      x_chr %in% c("0", "n", "no", "false", "f"), 0,
      suppressWarnings(as.numeric(x_chr))
    )
  )
  
  as.numeric(out)
}

impute_numeric <- function(x) {
  med <- median(x, na.rm = TRUE)
  if (is.na(med)) med <- 0
  x[is.na(x)] <- med
  x
}

impute_categorical <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Missing"
  as.factor(x)
}

fit_xgb_binary <- function(X, y, nrounds = 150) {
  dtrain <- xgb.DMatrix(data = X, label = y)
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
}

clip_probs <- function(p, lower = 0.05, upper = 0.95) {
  pmin(pmax(p, lower), upper)
}

# Doubly robust policy value for binary treatment
# Y = outcome
# W = historical treatment
# e = propensity P(W = 1 | X)
# m1 = predicted outcome if treated
# m0 = predicted outcome if untreated
# pi = target policy, 1 = treat, 0 = do not treat
dr_policy_value <- function(Y, W, e, m1, m0, pi) {
  e <- clip_probs(e)
  
  value_i <- pi * (m1 + (W / e) * (Y - m1)) +
    (1 - pi) * (m0 + ((1 - W) / (1 - e)) * (Y - m0))
  
  mean(value_i, na.rm = TRUE)
}

# ============================================================
# READ DATA
# ============================================================

df_raw <- read_excel(file_path)
names(df_raw) <- clean_names_simple(names(df_raw))

df <- as.data.frame(df_raw)

df$outcome_ed_90d <- to_binary(df$outcome_ed_90d)
df$intervention_flag <- to_binary(df$intervention_flag)

cat("Outcome distribution:\n")
print(table(df$outcome_ed_90d, useNA = "ifany"))

cat("Treatment distribution:\n")
print(table(df$intervention_flag, useNA = "ifany"))

# ============================================================
# DATE FEATURES
# ============================================================

date_fields <- c("index_date", "intervention_start_date", "intervention_end_date")

for (d in date_fields) {
  if (d %in% names(df)) {
    df[[d]] <- suppressWarnings(as.Date(df[[d]]))
  }
}

if ("intervention_start_date" %in% names(df)) {
  df$intervention_start_month <- month(df$intervention_start_date)
  df$intervention_start_wday <- wday(df$intervention_start_date)
} else {
  df$intervention_start_month <- NA_real_
  df$intervention_start_wday <- NA_real_
}

if (all(c("index_date", "intervention_start_date") %in% names(df))) {
  df$days_to_intervention_start <- as.numeric(df$intervention_start_date - df$index_date)
} else {
  df$days_to_intervention_start <- NA_real_
}

# ============================================================
# PREDICTORS
# ============================================================

predictor_vars <- c(
  "client_contract",
  "service_region",
  "program",
  "case_manager_name",
  "age",
  "gender",
  "dual_eligible",
  "county",
  "plan_type",
  "language",
  "living_alone_flag",
  
  "diabetes_flag",
  "chf_flag",
  "copd_flag",
  "asthma_flag",
  "depression_flag",
  "anxiety_flag",
  "substance_use_flag",
  "ckd_flag",
  "pregnancy_flag",
  "behavioral_health_risk_flag",
  
  "food_insecurity_flag",
  "housing_instability_flag",
  "transportation_barrier_flag",
  "utilities_insecurity_flag",
  
  "pcp_visits_last_6m",
  "specialist_visits_last_6m",
  "ed_visits_last_30d",
  "ed_visits_last_6m",
  "admits_last_6m",
  "observation_stays_last_6m",
  "total_cost_last_6m",
  "rx_count_last_6m",
  "med_adherence_pdc",
  "high_cost_drug_flag",
  "opioid_flag",
  "polypharmacy_flag",
  
  "percolator_utilization_score",
  "percolator_clinical_score",
  "percolator_sdoh_score",
  "current_risk_score",
  "risk_tier",
  
  # Outreach / engagement fields included
  "intervention_type",
  "intervention_days_active",
  "touches_per_month",
  "outreach_attempts",
  "successful_contacts",
  "avg_call_duration_min",
  "max_call_duration_min",
  "notes_escalation_flag",
  "community_referral_flag",
  "pharmacy_review_flag",
  "engagement_level",
  "engaged",
  "opted_out",
  "engagement_length",
  
  "days_to_intervention_start",
  "intervention_start_month",
  "intervention_start_wday"
)

predictor_vars <- predictor_vars[predictor_vars %in% names(df)]

model_df <- df[, c("outcome_ed_90d", "intervention_flag", predictor_vars), drop = FALSE]

model_df <- model_df[
  !is.na(model_df$outcome_ed_90d) &
    !is.na(model_df$intervention_flag),
  ,
  drop = FALSE
]

# ============================================================
# DATA CLEANING
# ============================================================

flag_cols <- grep("_flag$", names(model_df), value = TRUE)

for (col in flag_cols) {
  model_df[[col]] <- to_binary(model_df[[col]])
}

binary_extra <- c("engaged", "opted_out", "dual_eligible")
binary_extra <- binary_extra[binary_extra %in% names(model_df)]

for (col in binary_extra) {
  model_df[[col]] <- to_binary(model_df[[col]])
}

numeric_vars <- c(
  "age",
  "pcp_visits_last_6m",
  "specialist_visits_last_6m",
  "ed_visits_last_30d",
  "ed_visits_last_6m",
  "admits_last_6m",
  "observation_stays_last_6m",
  "total_cost_last_6m",
  "rx_count_last_6m",
  "med_adherence_pdc",
  "percolator_utilization_score",
  "percolator_clinical_score",
  "percolator_sdoh_score",
  "current_risk_score",
  "intervention_days_active",
  "touches_per_month",
  "outreach_attempts",
  "successful_contacts",
  "avg_call_duration_min",
  "max_call_duration_min",
  "engagement_length",
  "days_to_intervention_start",
  "intervention_start_month",
  "intervention_start_wday"
)

numeric_vars <- numeric_vars[numeric_vars %in% names(model_df)]

for (col in numeric_vars) {
  model_df[[col]] <- suppressWarnings(as.numeric(model_df[[col]]))
}

for (col in names(model_df)) {
  if (col %in% c("outcome_ed_90d", "intervention_flag")) next
  
  if (is.numeric(model_df[[col]])) {
    model_df[[col]] <- impute_numeric(model_df[[col]])
  } else {
    model_df[[col]] <- impute_categorical(model_df[[col]])
  }
}

unique_counts <- sapply(model_df, function(x) length(unique(x[!is.na(x)])))
keep_cols <- names(unique_counts[unique_counts > 1])
model_df <- model_df[, keep_cols, drop = FALSE]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

set.seed(123)

n <- nrow(model_df)
train_idx <- sample(seq_len(n), size = floor(0.70 * n))
test_idx <- setdiff(seq_len(n), train_idx)

train_df <- model_df[train_idx, , drop = FALSE]
test_df <- model_df[test_idx, , drop = FALSE]

# ============================================================
# MODEL MATRICES
# ============================================================

x_vars <- setdiff(names(model_df), c("outcome_ed_90d", "intervention_flag"))

combined_x_df <- rbind(
  train_df[, x_vars, drop = FALSE],
  test_df[, x_vars, drop = FALSE]
)

combined_matrix <- model.matrix(~ . - 1, data = combined_x_df)

x_train <- combined_matrix[1:nrow(train_df), , drop = FALSE]
x_test <- combined_matrix[(nrow(train_df) + 1):nrow(combined_matrix), , drop = FALSE]

Y_train <- as.numeric(train_df$outcome_ed_90d)
W_train <- as.numeric(train_df$intervention_flag)

Y_test <- as.numeric(test_df$outcome_ed_90d)
W_test <- as.numeric(test_df$intervention_flag)

# ============================================================
# NUISANCE MODELS
# ============================================================

# 1. Propensity model: P(treatment | X)
propensity_model <- fit_xgb_binary(x_train, W_train)

# 2. Outcome model among treated: E[Y | X, W = 1]
treated_train <- train_df$intervention_flag == 1
control_train <- train_df$intervention_flag == 0

x_train_treated <- x_train[treated_train, , drop = FALSE]
y_train_treated <- Y_train[treated_train]

x_train_control <- x_train[control_train, , drop = FALSE]
y_train_control <- Y_train[control_train]

outcome_model_treated <- fit_xgb_binary(x_train_treated, y_train_treated)
outcome_model_control <- fit_xgb_binary(x_train_control, y_train_control)

# ============================================================
# PREDICT NUISANCE COMPONENTS ON TEST SET
# ============================================================

dtest <- xgb.DMatrix(data = x_test)

e_hat <- predict(propensity_model, dtest)
m1_hat <- predict(outcome_model_treated, dtest)
m0_hat <- predict(outcome_model_control, dtest)

e_hat <- clip_probs(e_hat)

benefit_score <- m0_hat - m1_hat

scored_test <- test_df %>%
  mutate(
    propensity_score = e_hat,
    pred_ed_if_treated = m1_hat,
    pred_ed_if_control = m0_hat,
    benefit_score = benefit_score
  )

scored_test$benefit_rank <- rank(-scored_test$benefit_score, ties.method = "first")
scored_test$benefit_percentile <- scored_test$benefit_rank / nrow(scored_test)

# ============================================================
# DEFINE TARGET POLICIES
# ============================================================

policy_historical <- W_test
policy_treat_none <- rep(0, length(W_test))
policy_treat_all <- rep(1, length(W_test))

policy_top_10 <- ifelse(scored_test$benefit_percentile <= 0.10, 1, 0)
policy_top_20 <- ifelse(scored_test$benefit_percentile <= 0.20, 1, 0)
policy_top_30 <- ifelse(scored_test$benefit_percentile <= 0.30, 1, 0)
policy_top_40 <- ifelse(scored_test$benefit_percentile <= 0.40, 1, 0)

# ============================================================
# DOUBLY ROBUST POLICY EVALUATION
# ============================================================

policy_results <- data.frame(
  policy = c(
    "Historical observed policy",
    "Treat nobody",
    "Treat everybody",
    "Treat top 10% by benefit score",
    "Treat top 20% by benefit score",
    "Treat top 30% by benefit score",
    "Treat top 40% by benefit score"
  ),
  treatment_rate = c(
    mean(policy_historical),
    mean(policy_treat_none),
    mean(policy_treat_all),
    mean(policy_top_10),
    mean(policy_top_20),
    mean(policy_top_30),
    mean(policy_top_40)
  ),
  estimated_ed_rate = c(
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_historical),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_treat_none),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_treat_all),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_top_10),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_top_20),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_top_30),
    dr_policy_value(Y_test, W_test, e_hat, m1_hat, m0_hat, policy_top_40)
  )
)

historical_ed_rate <- policy_results$estimated_ed_rate[
  policy_results$policy == "Historical observed policy"
]

policy_results <- policy_results %>%
  mutate(
    estimated_ed_rate_reduction_vs_historical = historical_ed_rate - estimated_ed_rate,
    expected_ed_visits_avoided_per_1000 = estimated_ed_rate_reduction_vs_historical * 1000
  ) %>%
  arrange(estimated_ed_rate)

print(policy_results)

# ============================================================
# OPTIONAL ROI ASSUMPTIONS
# ============================================================

cost_per_ed_visit <- 1200
cost_per_intervention <- 250

policy_results <- policy_results %>%
  mutate(
    expected_ed_savings_per_1000 = expected_ed_visits_avoided_per_1000 * cost_per_ed_visit,
    intervention_cost_per_1000 = treatment_rate * 1000 * cost_per_intervention,
    net_savings_per_1000 = expected_ed_savings_per_1000 - intervention_cost_per_1000
  )

print(policy_results)

# ============================================================
# SAVE OUTPUTS
# ============================================================

write_csv(policy_results, output_path)
write_csv(scored_test, scored_output_path)

cat("Policy evaluation output saved to:\n", output_path, "\n\n")
cat("Scored member output saved to:\n", scored_output_path, "\n\n")

cat("INTERPRETATION:\n")
cat("- estimated_ed_rate = doubly robust estimated ED rate under that policy\n")
cat("- Lower estimated_ed_rate is better\n")
cat("- expected_ed_visits_avoided_per_1000 compares each policy to historical targeting\n")
cat("- net_savings_per_1000 applies your cost assumptions\n")