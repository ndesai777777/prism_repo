# ============================================================
# T-LEARNER UPLIFT MODEL WITH XGBOOST
# Outcome: outcome_ed_90d
# Treatment: intervention_flag
# File type: Excel .xlsx
# ============================================================

required_packages <- c("readxl", "readr", "dplyr", "lubridate", "xgboost")

installed <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg)
  }
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

output_path <- "D:/Users/ben.novinger/OneDrive - Acentra/Analytics Opportunities Workgroup/AI Innovation Challenge/t_learner_scored_output.csv"

summary_path <- "D:/Users/ben.novinger/OneDrive - Acentra/Analytics Opportunities Workgroup/AI Innovation Challenge/t_learner_decile_summary.csv"

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

# ============================================================
# READ FILE
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
  # demographics / cohort
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
  
  # clinical
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
  
  # SDOH
  "food_insecurity_flag",
  "housing_instability_flag",
  "transportation_barrier_flag",
  "utilities_insecurity_flag",
  
  # prior utilization / pharmacy
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
  
  # risk scores
  "percolator_utilization_score",
  "percolator_clinical_score",
  "percolator_sdoh_score",
  "current_risk_score",
  "risk_tier",
  
  # intervention / outreach / engagement fields
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
  
  # derived timing
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

train_df <- model_df[train_idx, , drop = FALSE]
test_df <- model_df[-train_idx, , drop = FALSE]

train_treated <- train_df[train_df$intervention_flag == 1, , drop = FALSE]
train_control <- train_df[train_df$intervention_flag == 0, , drop = FALSE]

cat("Training treated rows:", nrow(train_treated), "\n")
cat("Training control rows:", nrow(train_control), "\n")

# ============================================================
# MODEL MATRICES
# ============================================================

train_treated_x_df <- train_treated[, setdiff(names(train_treated), c("outcome_ed_90d", "intervention_flag")), drop = FALSE]
train_control_x_df <- train_control[, setdiff(names(train_control), c("outcome_ed_90d", "intervention_flag")), drop = FALSE]
test_x_df <- test_df[, setdiff(names(test_df), c("outcome_ed_90d", "intervention_flag")), drop = FALSE]

combined_x_df <- rbind(train_treated_x_df, train_control_x_df, test_x_df)
combined_matrix <- model.matrix(~ . - 1, data = combined_x_df)

n_treated <- nrow(train_treated_x_df)
n_control <- nrow(train_control_x_df)
n_test <- nrow(test_x_df)

x_treated <- combined_matrix[1:n_treated, , drop = FALSE]
x_control <- combined_matrix[(n_treated + 1):(n_treated + n_control), , drop = FALSE]
x_test <- combined_matrix[(n_treated + n_control + 1):(n_treated + n_control + n_test), , drop = FALSE]

y_treated <- as.numeric(train_treated$outcome_ed_90d)
y_control <- as.numeric(train_control$outcome_ed_90d)

# ============================================================
# TRAIN T-LEARNER MODELS
# ============================================================

dtrain_treated <- xgb.DMatrix(data = x_treated, label = y_treated)
dtrain_control <- xgb.DMatrix(data = x_control, label = y_control)
dtest <- xgb.DMatrix(data = x_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(123)
model_treated <- xgb.train(
  params = params,
  data = dtrain_treated,
  nrounds = 150,
  verbose = 0
)

set.seed(123)
model_control <- xgb.train(
  params = params,
  data = dtrain_control,
  nrounds = 150,
  verbose = 0
)

cat("T-Learner models trained successfully.\n")

# ============================================================
# SCORE TEST SET
# ============================================================

p_treated <- predict(model_treated, dtest)
p_control <- predict(model_control, dtest)

results_test <- test_df %>%
  mutate(
    pred_ed_if_treated = p_treated,
    pred_ed_if_control = p_control,
    
    # ED is bad, so positive benefit means treatment reduces ED risk
    benefit_score = pred_ed_if_control - pred_ed_if_treated,
    
    # Treatment effect on bad outcome
    treatment_effect_bad_outcome = pred_ed_if_treated - pred_ed_if_control
  )

results_test$uplift_decile <- dplyr::ntile(dplyr::desc(results_test$benefit_score), 10)

# ============================================================
# DECILE SUMMARY
# ============================================================

decile_summary <- results_test %>%
  group_by(uplift_decile) %>%
  summarise(
    n = n(),
    avg_benefit_score = mean(benefit_score, na.rm = TRUE),
    observed_ed_rate = mean(outcome_ed_90d, na.rm = TRUE),
    treated_pct = mean(intervention_flag, na.rm = TRUE),
    avg_pred_ed_if_treated = mean(pred_ed_if_treated, na.rm = TRUE),
    avg_pred_ed_if_control = mean(pred_ed_if_control, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(uplift_decile)

print(decile_summary)

# ============================================================
# SCORE FULL FILE
# ============================================================

full_x_df <- model_df[, setdiff(names(model_df), c("outcome_ed_90d", "intervention_flag")), drop = FALSE]
full_matrix <- model.matrix(~ . - 1, data = full_x_df)

missing_in_full <- setdiff(colnames(combined_matrix), colnames(full_matrix))

if (length(missing_in_full) > 0) {
  for (col in missing_in_full) {
    full_matrix <- cbind(full_matrix, 0)
    colnames(full_matrix)[ncol(full_matrix)] <- col
  }
}

extra_in_full <- setdiff(colnames(full_matrix), colnames(combined_matrix))

if (length(extra_in_full) > 0) {
  full_matrix <- full_matrix[, !(colnames(full_matrix) %in% extra_in_full), drop = FALSE]
}

full_matrix <- full_matrix[, colnames(combined_matrix), drop = FALSE]

dfull <- xgb.DMatrix(data = full_matrix)

full_pred_treated <- predict(model_treated, dfull)
full_pred_control <- predict(model_control, dfull)

scored_full <- model_df %>%
  mutate(
    pred_ed_if_treated = full_pred_treated,
    pred_ed_if_control = full_pred_control,
    benefit_score = pred_ed_if_control - pred_ed_if_treated,
    treatment_effect_bad_outcome = pred_ed_if_treated - pred_ed_if_control
  )

scored_full$uplift_decile <- dplyr::ntile(dplyr::desc(scored_full$benefit_score), 10)

# ============================================================
# SAVE OUTPUTS
# ============================================================

write_csv(scored_full, output_path)
write_csv(decile_summary, summary_path)

cat("Scored output saved to:\n", output_path, "\n")
cat("Decile summary saved to:\n", summary_path, "\n")

# ============================================================
# INTERPRETATION
# ============================================================

cat("\nINTERPRETATION:\n")
cat("- pred_ed_if_treated = predicted ED probability if treated\n")
cat("- pred_ed_if_control = predicted ED probability if untreated\n")
cat("- benefit_score = pred_ed_if_control - pred_ed_if_treated\n")
cat("- Higher benefit_score = greater expected ED reduction from intervention\n")
cat("- uplift_decile 1 = highest expected treatment benefit\n")