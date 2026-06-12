# ============================================================
# CAUSAL FOREST MODEL USING grf
# Outcome: outcome_ed_90d
# Treatment: intervention_flag
# Input: Excel (.xlsx)
# ============================================================

required_packages <- c("readxl", "readr", "dplyr", "lubridate", "grf")

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
library(grf)

# ============================================================
# FILE PATHS
# ============================================================

github_xlsx_url <- "https://raw.githubusercontent.com/ndesai777777/prism_repo/main/DataSets/PRP_1000_full_pretreatment.xlsx"

temp_xlsx <- tempfile(fileext = ".xlsx")
download.file(github_xlsx_url, destfile = temp_xlsx, mode = "wb")

file_path <- temp_xlsx

output_folder <- "Outputs/Causal-Forests"
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(output_folder, "causal_forest_scored_output.csv")
summary_path <- file.path(output_folder, "causal_forest_decile_summary.csv")
importance_path <- file.path(output_folder, "causal_forest_variable_importance.csv")

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
# READ DATA
# ============================================================

df_raw <- read_excel(file_path)
names(df_raw) <- clean_names_simple(names(df_raw))

cat("Rows:", nrow(df_raw), "\n")
cat("Columns:", ncol(df_raw), "\n\n")
print(names(df_raw))

df <- as.data.frame(df_raw)

# ============================================================
# REQUIRED FIELDS
# ============================================================

required_fields <- c("outcome_ed_90d", "intervention_flag")

missing_required <- setdiff(required_fields, names(df))

if (length(missing_required) > 0) {
  stop(paste("Missing required columns:", paste(missing_required, collapse = ", ")))
}

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

if (all(c("intervention_start_date", "intervention_end_date") %in% names(df))) {
  df$intervention_duration_calc <- as.numeric(df$intervention_end_date - df$intervention_start_date)
} else {
  df$intervention_duration_calc <- NA_real_
}

if (!("intervention_days_active" %in% names(df))) {
  df$intervention_days_active <- df$intervention_duration_calc
}

# ============================================================
# SIMPLIFIED PREDICTORS (FIX FOR CAUSAL FOREST)
# ============================================================

predictor_vars <- c(
  
  # Demographics
  "age",
  "gender",
  "dual_eligible",
  
  # Clinical flags
  "diabetes_flag",
  "chf_flag",
  "copd_flag",
  "asthma_flag",
  "depression_flag",
  "anxiety_flag",
  "substance_use_flag",
  "ckd_flag",
  
  # SDOH
  "food_insecurity_flag",
  "housing_instability_flag",
  "transportation_barrier_flag",
  
  # Utilization
  "ed_visits_last_30d",
  "ed_visits_last_6m",
  "admits_last_6m",
  "total_cost_last_6m",
  
  # Risk scores
  "percolator_utilization_score",
  "percolator_clinical_score",
  "current_risk_score",
  
  # Outreach / engagement (KEEP THESE per your request)
  "touches_per_month",
  "outreach_attempts",
  "successful_contacts",
  "avg_call_duration_min",
  "community_referral_flag",
  "pharmacy_review_flag",
  "engagement_level"
)

# ============================================================
# DATA TYPE HANDLING
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

# Remove columns with only one value
unique_counts <- sapply(model_df, function(x) length(unique(x[!is.na(x)])))
keep_cols <- names(unique_counts[unique_counts > 1])
model_df <- model_df[, keep_cols, drop = FALSE]

# ============================================================
# CREATE MODEL MATRIX
# ============================================================

Y <- as.numeric(model_df$outcome_ed_90d)
W <- as.numeric(model_df$intervention_flag)

X_df <- model_df[, setdiff(names(model_df), c("outcome_ed_90d", "intervention_flag")), drop = FALSE]

X <- model.matrix(~ . - 1, data = X_df)

cat("Final model matrix dimensions:\n")
print(dim(X))

if (!all(Y %in% c(0, 1))) {
  stop("Outcome contains values other than 0 and 1.")
}

if (!all(W %in% c(0, 1))) {
  stop("Treatment contains values other than 0 and 1.")
}

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

set.seed(123)

n <- nrow(X)
train_idx <- sample(seq_len(n), size = floor(0.70 * n))
test_idx <- setdiff(seq_len(n), train_idx)

X_train <- X[train_idx, , drop = FALSE]
Y_train <- Y[train_idx]
W_train <- W[train_idx]

X_test <- X[test_idx, , drop = FALSE]
Y_test <- Y[test_idx]
W_test <- W[test_idx]

# ============================================================
# FIT CAUSAL FOREST
# ============================================================

set.seed(123)

cf_model <- causal_forest(
  X = X_train,
  Y = Y_train,
  W = W_train,
  num.trees = 2000,
  honesty = TRUE,
  seed = 123
)

cat("Causal forest trained successfully.\n\n")

# Average treatment effect
ate <- average_treatment_effect(cf_model)

cat("Average Treatment Effect estimate:\n")
print(ate)
cat("\n")

# ============================================================
# PREDICT INDIVIDUAL TREATMENT EFFECTS
# ============================================================

cf_pred_test <- predict(cf_model, X_test, estimate.variance = TRUE)

results_test <- model_df[test_idx, , drop = FALSE] %>%
  mutate(
    tau_hat = as.numeric(cf_pred_test$predictions),
    tau_se = sqrt(as.numeric(cf_pred_test$variance.estimates)),
    
    # Since outcome_ed_90d is bad:
    # negative tau means intervention reduces ED risk.
    benefit_score = -tau_hat,
    
    uplift_decile = dplyr::ntile(dplyr::desc(benefit_score), 10)
  )

# ============================================================
# DECILE SUMMARY
# ============================================================

decile_summary <- results_test %>%
  group_by(uplift_decile) %>%
  summarise(
    n = n(),
    avg_tau_hat = mean(tau_hat, na.rm = TRUE),
    avg_benefit_score = mean(benefit_score, na.rm = TRUE),
    observed_ed_rate = mean(outcome_ed_90d, na.rm = TRUE),
    treated_pct = mean(intervention_flag, na.rm = TRUE),
    avg_tau_se = mean(tau_se, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(uplift_decile)

cat("Decile summary:\n")
print(decile_summary)

# ============================================================
# VARIABLE IMPORTANCE
# ============================================================

var_imp <- variable_importance(cf_model)

importance_df <- data.frame(
  feature = colnames(X_train),
  importance = as.numeric(var_imp)
) %>%
  arrange(desc(importance))

cat("Top variable importance:\n")
print(head(importance_df, 25))

# ============================================================
# SCORE FULL FILE
# ============================================================

cf_pred_full <- predict(cf_model, X, estimate.variance = TRUE)

scored_full <- model_df %>%
  mutate(
    tau_hat = as.numeric(cf_pred_full$predictions),
    tau_se = sqrt(as.numeric(cf_pred_full$variance.estimates)),
    benefit_score = -tau_hat,
    uplift_decile = dplyr::ntile(dplyr::desc(benefit_score), 10)
  )

# ============================================================
# SAVE OUTPUTS
# ============================================================

write_csv(scored_full, output_path)
write_csv(decile_summary, summary_path)
write_csv(importance_df, importance_path)

cat("Scored output saved to:\n", output_path, "\n\n")
cat("Decile summary saved to:\n", summary_path, "\n\n")
cat("Variable importance saved to:\n", importance_path, "\n\n")

# ============================================================
# INTERPRETATION
# ============================================================

cat("INTERPRETATION:\n")
cat("- tau_hat = estimated treatment effect on outcome_ed_90d\n")
cat("- Because outcome_ed_90d is bad, negative tau_hat means intervention reduced ED risk\n")
cat("- benefit_score = -tau_hat\n")
cat("- Higher benefit_score means larger estimated ED reduction from intervention\n")
cat("- uplift_decile 1 = highest estimated intervention benefit\n")