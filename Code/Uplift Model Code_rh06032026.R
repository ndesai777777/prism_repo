# ============================================================
# UPLIFT MODEL SCRIPT (T-LEARNER WITH XGBOOST)
# Reads from Excel (.xlsx)
# Outcome: outcome_ed_90d
# Treatment: intervention_flag
# ============================================================

#### TODO: Some concerns: 
#### Cross validation may work better than 70/30 split. 
#### No performance check on individual XGboost models
#### Compare performance of other ML algorithms, let's use RGLM (glmnet) as benchmark.
#### If resource and data security allowed, we can also try automl which includes all major ML algorithms.

# ----------------------------
# 1. Install packages if needed
# ----------------------------
required_packages <- c("readxl", "readr", "dplyr", "stringr", "lubridate", "xgboost","ggplot2","tidyr",
                       "glmnet","pROC")

installed <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg)
  }
}

# ----------------------------
# 2. Load packages
# ----------------------------
library(readxl)
library(readr)
library(dplyr)
library(stringr)
library(lubridate)
library(xgboost)

# ============================================================
# 3. FILE PATHS
# ============================================================

file_path <- "C:/Users/Rui.Huang/OneDrive - Acentra/Documents/PRISM/prism_simulated_1000_full_pretreatment.xlsx"
output_path <- "D:/Users/Rui.Huang/OneDrive - Acentra/Documents/PRISM/uplift_scored_output.csv"
summary_path <- "D:/Users/Rui.Huang/OneDrive - Acentra/Documents/PRISM/uplift_decile_summary.csv"

cat("Input file path:\n", file_path, "\n\n")

if (!file.exists(file_path)) {
  stop("File not found. Check the file path.")
}

# ============================================================
# 4. HELPER FUNCTIONS
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
} ## TODO: The output of this function is numeric. Is it intentionally for xgboost? Techinically it should work for xgboost
#### But if later we explore other algorithms it may be problematic.

safe_as_date <- function(x) {
  suppressWarnings(as.Date(x))
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
# 5. READ EXCEL FILE
# ============================================================

df_raw <- read_excel(file_path)
names(df_raw) <- clean_names_simple(names(df_raw))

cat("Rows:", nrow(df_raw), "\n")
cat("Columns:", ncol(df_raw), "\n\n")

cat("Column names after cleaning:\n")
print(names(df_raw))
cat("\n")

# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_fields <- c("outcome_ed_90d", "intervention_flag")

missing_required <- setdiff(required_fields, names(df_raw))
if (length(missing_required) > 0) {
  stop(paste("Missing required columns:", paste(missing_required, collapse = ", ")))
}

# ============================================================
# 7. BASIC CLEANUP
# ============================================================

df <- as.data.frame(df_raw)

date_fields <- c("index_date", "intervention_start_date", "intervention_end_date")
for (d in date_fields) {
  if (d %in% names(df)) {
    df[[d]] <- safe_as_date(df[[d]])
  }
}

df$intervention_flag <- to_binary(df$intervention_flag)
df$outcome_ed_90d <- to_binary(df$outcome_ed_90d)  ### Still numeric here

# ============================================================
# 8. DERIVE DATE FEATURES
# ============================================================

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
## reverse for now
df = df_raw

# ============================================================
# 9. SELECT PREDICTORS
# ============================================================

candidate_predictors <- c(
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
  "days_to_intervention_start",
  "intervention_start_month",
  "intervention_start_wday"
)

candidate_predictors <- candidate_predictors[candidate_predictors %in% names(df)]
missing_predictors <- setdiff(candidate_predictors, names(df))

### TODO: Added information about missing candidate_predictors 
if (length(missing_predictors) > 0) {
  cat("Predictors not found in dataset:\n")
  print(missing_predictors)
} else {
  cat("All candidate predictors are present in dataset.\n")
}

model_df <- df[, c("outcome_ed_90d", "intervention_flag", candidate_predictors), drop = FALSE]
model_df <- model_df[!is.na(model_df$outcome_ed_90d) & !is.na(model_df$intervention_flag), , drop = FALSE]

### TODO: Added information about unselected columns
missing_in_model_df <- setdiff(names(df), names(model_df))

if (length(missing_in_model_df) > 0) {
  cat("Columns in dataframe but not in model:\n")
  print(missing_in_model_df)
} else {
  cat("All dataframe columns are present in model.\n")
}
cat("Modeling rows after dropping missing outcome/treatment:", nrow(model_df), "\n\n")

# ============================================================
# 10. DATA TYPE HANDLING
# ============================================================

flag_like_cols <- grep("_flag$", names(model_df), value = TRUE)

for (col in flag_like_cols) {
  model_df[[col]] <- to_binary(model_df[[col]])
}

possible_numeric_cols <- c(
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

possible_numeric_cols <- possible_numeric_cols[possible_numeric_cols %in% names(model_df)]

for (col in possible_numeric_cols) {
  model_df[[col]] <- suppressWarnings(as.numeric(model_df[[col]]))
}

for (col in names(model_df)) {
  if (!(col %in% c("outcome_ed_90d", "intervention_flag"))) {
    if (!is.numeric(model_df[[col]])) {
      model_df[[col]] <- impute_categorical(model_df[[col]])
    }
  }
}

for (col in names(model_df)) {
  if (col %in% c("outcome_ed_90d", "intervention_flag")) next
  if (is.numeric(model_df[[col]])) {
    model_df[[col]] <- impute_numeric(model_df[[col]])
  }
}

unique_counts <- sapply(model_df, function(x) length(unique(x[!is.na(x)])))
keep_cols <- names(unique_counts[unique_counts > 1])
model_df <- model_df[, keep_cols, drop = FALSE]

cat("Final modeling columns:\n")
print(names(model_df))
cat("\n")

# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

set.seed(123)

n <- nrow(model_df)
train_idx <- sample(seq_len(n), size = floor(0.70 * n))

train_df <- model_df[train_idx, , drop = FALSE]
test_df <- model_df[-train_idx, , drop = FALSE]

cat("Training rows:", nrow(train_df), "\n")
cat("Testing rows:", nrow(test_df), "\n\n")
### TODO: Might also consider cross validation rather than random split

# ============================================================
# 12. SEPARATE TREATED / CONTROL
# ============================================================

train_treated <- train_df[train_df$intervention_flag == 1, , drop = FALSE]
train_control <- train_df[train_df$intervention_flag == 0, , drop = FALSE]

cat("Training treated rows:", nrow(train_treated), "\n")
cat("Training control rows:", nrow(train_control), "\n\n")

if (nrow(train_treated) < 50) stop("Too few treated rows to train a stable model.")
if (nrow(train_control) < 50) stop("Too few control rows to train a stable model.")

# ============================================================
# 13. BUILD MODEL MATRICES
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

y_treated <- train_treated$outcome_ed_90d
y_control <- train_control$outcome_ed_90d

# ============================================================
# 14. TRAIN XGBOOST MODELS
# ============================================================

# Make absolutely sure the target is numeric 0/1
y_treated <- as.numeric(y_treated)
y_control <- as.numeric(y_control)

cat("Unique y_treated values:\n")
print(sort(unique(y_treated)))
cat("\n")

cat("Unique y_control values:\n")
print(sort(unique(y_control)))
cat("\n")

if (!all(y_treated %in% c(0, 1))) {
  stop("y_treated contains values other than 0 and 1.")
}

if (!all(y_control %in% c(0, 1))) {
  stop("y_control contains values other than 0 and 1.")
}

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
  nrounds = 150
)

set.seed(123)
model_control <- xgb.train(
  params = params,
  data = dtrain_control,
  nrounds = 150,
  verbose = 0
)

cat("Models trained successfully.\n\n")

##### TODO: No evaluation of the performance of these models

#### Calculate AUC from the results
library(pROC)

ind_test_treated <- which(test_df$intervention_flag == 1)
ind_test_control <- which(test_df$intervention_flag == 0)

pred_treated <- predict(model_treated, dtest[ind_test_treated,])

roc_treated <- roc(test_df$outcome_ed_90d[ind_test_treated], pred_treated)

auc_treated <- auc(roc_treated)

cat("XGBoost Treated model AUC:", round(as.numeric(auc_treated), 4), "\n")


pred_control <- predict(model_control, dtest[ind_test_control,])

roc_control <- roc(test_df$outcome_ed_90d[ind_test_control], pred_control)

auc_control <- auc(roc_control)

cat("XGBoost Control model AUC:", round(as.numeric(auc_control), 4), "\n")


#### Actually we should do crossvalidation xgboost to search for hyperparameter first, instead of using preset hyperparameters


# ============================================================
# XGBOOST CV GRID SEARCH FUNCTION
# ============================================================

fit_xgb_cv_grid <- function(dtrain, grid, nrounds_max = 500, nfold = 5) {
  
  results <- data.frame()
  best_model_info <- NULL
  best_auc <- -Inf
  
  for (i in seq_len(nrow(grid))) {
    
    params_i <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = grid$max_depth[i],
      eta = grid$eta[i],
      min_child_weight = grid$min_child_weight[i],
      subsample = 0.8,
      colsample_bytree = 0.8
    )
    
    set.seed(123)
    cv_i <- xgb.cv(
      params = params_i,
      data = dtrain,
      nrounds = nrounds_max,
      nfold = nfold,
      early_stopping_rounds = 20,
      verbose = 0
    )
    
    best_iter_i <- which.max(cv_i$evaluation_log$test_auc_mean)
    best_auc_i <- cv_i$evaluation_log$test_auc_mean[best_iter_i]
    
    results <- rbind(
      results,
      data.frame(
        max_depth = grid$max_depth[i],
        eta = grid$eta[i],
        min_child_weight = grid$min_child_weight[i],
        best_nrounds = best_iter_i,
        cv_auc = best_auc_i
      )
    )
    
    if (best_auc_i > best_auc) {
      best_auc <- best_auc_i
      best_model_info <- list(
        params = params_i,
        best_nrounds = best_iter_i,
        cv_auc = best_auc_i,
        cv_object = cv_i
      )
    }
  }
  
  results <- results[order(-results$cv_auc), ]
  
  final_model <- xgb.train(
    params = best_model_info$params,
    data = dtrain,
    nrounds = best_model_info$best_nrounds,
    verbose = 0
  )
  
  list(
    model = final_model,
    best_params = best_model_info$params,
    best_nrounds = best_model_info$best_nrounds,
    best_cv_auc = best_model_info$cv_auc,
    search_results = results
  )
}


# ============================================================
# CREATE DMATRICES
# ============================================================

dtrain_treated <- xgb.DMatrix(data = x_treated, label = y_treated)
dtrain_control <- xgb.DMatrix(data = x_control, label = y_control)
dtest <- xgb.DMatrix(data = x_test)

# ============================================================
# SMALL XGBOOST GRID
# ============================================================

xgb_grid <- expand.grid(
  max_depth = c(3, 4, 5),
  eta = c(0.03, 0.05, 0.10),
  min_child_weight = c(1, 5)
)

# ============================================================
# TRAIN TREATED MODEL WITH CV GRID SEARCH
# ============================================================

xgb_treated_cv <- fit_xgb_cv_grid(
  dtrain = dtrain_treated,
  grid = xgb_grid,
  nrounds_max = 500,
  nfold = 5
)

model_treated <- xgb_treated_cv$model

cat("XGBoost Treated best CV AUC:",
    round(xgb_treated_cv$best_cv_auc, 4), "\n")
cat("XGBoost Treated best nrounds:",
    xgb_treated_cv$best_nrounds, "\n")
cat("XGBoost Treated best params:\n")
print(xgb_treated_cv$best_params)

# ============================================================
# TRAIN CONTROL MODEL WITH CV GRID SEARCH
# ============================================================

xgb_control_cv <- fit_xgb_cv_grid(
  dtrain = dtrain_control,
  grid = xgb_grid,
  nrounds_max = 500,
  nfold = 5
)

model_control <- xgb_control_cv$model

cat("XGBoost Control best CV AUC:",
    round(xgb_control_cv$best_cv_auc, 4), "\n")
cat("XGBoost Control best nrounds:",
    xgb_control_cv$best_nrounds, "\n")
cat("XGBoost Control best params:\n")
print(xgb_control_cv$best_params)

# ============================================================
# TEST AUC FOR CV-TUNED XGBOOST MODELS
# ============================================================

ind_test_treated <- which(test_df$intervention_flag == 1)
ind_test_control <- which(test_df$intervention_flag == 0)

pred_treated_cv_xgb <- predict(model_treated, dtest[ind_test_treated, ])
roc_treated_cv_xgb <- roc(test_df$outcome_ed_90d[ind_test_treated], pred_treated_cv_xgb)
auc_treated_cv_xgb <- auc(roc_treated_cv_xgb)

cat("XGBoost Treated CV-tuned test AUC:",
    round(as.numeric(auc_treated_cv_xgb), 4), "\n")

pred_control_cv_xgb <- predict(model_control, dtest[ind_test_control, ])
roc_control_cv_xgb <- roc(test_df$outcome_ed_90d[ind_test_control], pred_control_cv_xgb)
auc_control_cv_xgb <- auc(roc_control_cv_xgb)

cat("XGBoost Control CV-tuned test AUC:",
    round(as.numeric(auc_control_cv_xgb), 4), "\n")

cat("CV-tuned XGBoost models trained successfully.\n\n")




#### Now Add GLMNET


library(glmnet)

# ============================================================
# TRAIN GLMNET MODELS
# ============================================================


# ============================================================
# ELASTIC NET HYPERPARAMETER SEARCH
# ============================================================

fit_elastic_net <- function(x, y,
                            alpha_grid = seq(0, 1, by = 0.1),
                            nfolds = 5) {
  
  results <- data.frame()
  
  best_model <- NULL
  best_auc <- -Inf
  best_alpha <- NA
  
  for (a in alpha_grid) {
    
    set.seed(123)
    
    cv_fit <- cv.glmnet(
      x = x,
      y = y,
      family = "binomial",
      alpha = a,
      nfolds = nfolds,
      type.measure = "auc"
    )
    
    auc_cv <- max(cv_fit$cvm)
    
    results <- rbind(
      results,
      data.frame(
        alpha = a,
        lambda = cv_fit$lambda.min,
        cv_auc = auc_cv
      )
    )
    
    if (auc_cv > best_auc) {
      best_auc <- auc_cv
      best_alpha <- a
      best_model <- cv_fit
    }
  }
  
  list(
    best_model = best_model,
    best_alpha = best_alpha,
    best_lambda = best_model$lambda.min,
    best_auc = best_auc,
    search_results = results[order(-results$cv_auc), ]
  )
}

# ============================================================
# TREATED MODEL
# ============================================================

enet_treated <- fit_elastic_net(
  x = x_treated,
  y = y_treated
)

cat("Best treated alpha:", enet_treated$best_alpha, "\n")
cat("Best treated lambda:", enet_treated$best_lambda, "\n")
cat("Best treated CV AUC:", round(enet_treated$best_auc, 4), "\n\n")

# ============================================================
# CONTROL MODEL
# ============================================================

enet_control <- fit_elastic_net(
  x = x_control,
  y = y_control
)

cat("Best control alpha:", enet_control$best_alpha, "\n")
cat("Best control lambda:", enet_control$best_lambda, "\n")
cat("Best control CV AUC:", round(enet_control$best_auc, 4), "\n\n")


# ============================================================
# TEST AUC FOR CV-TUNED GLMNET MODELS
# ============================================================

pred_treated_cv_glmnet <- as.numeric(
  predict(enet_treated$best_model,
          newx = x_test[ind_test_treated, , drop = FALSE],
          s = "lambda.min",
          type = "response")
)
roc_treated_cv_glmnet <- roc(test_df$outcome_ed_90d[ind_test_treated], pred_treated_cv_glmnet)
auc_treated_cv_glmnet <- auc(roc_treated_cv_glmnet)

cat("GLMNET Treated CV-tuned test AUC:",
    round(as.numeric(auc_treated_cv_glmnet), 4), "\n")

pred_control_cv_glmnet <- as.numeric(
  predict(enet_control$best_model,
          newx = x_test[ind_test_control, , drop = FALSE],
          s = "lambda.min",
          type = "response")
)
roc_control_cv_glmnet <- roc(test_df$outcome_ed_90d[ind_test_control], pred_control_cv_glmnet)
auc_control_cv_glmnet <- auc(roc_control_cv_glmnet)

cat("GLMNET Control CV-tuned test AUC:",
    round(as.numeric(auc_control_cv_glmnet), 4), "\n\n")



# ============================================================
# 15. SCORE TEST SET
# ============================================================

p_treated <- predict(model_treated, dtest)
p_control <- predict(model_control, dtest)

results_test <- test_df %>%
  mutate(
    pred_ed_if_treated = p_treated,
    pred_ed_if_control = p_control,
    benefit_score = pred_ed_if_control - pred_ed_if_treated,
    uplift_bad_outcome = pred_ed_if_treated - pred_ed_if_control
  )

results_test$uplift_decile <- dplyr::ntile(dplyr::desc(results_test$benefit_score), 10)

cat("Top 20 highest-benefit members:\n")
print(
  results_test %>%
    arrange(desc(benefit_score)) %>%
    select(outcome_ed_90d, intervention_flag, pred_ed_if_treated, pred_ed_if_control, benefit_score, uplift_decile) %>%
    head(20)
)
cat("\n")

# ============================================================
# 16. DECILE SUMMARY
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

cat("Decile summary:\n")
print(decile_summary)
cat("\n")

# ============================================================
# 17. VARIABLE IMPORTANCE
# ============================================================

importance_treated <- xgb.importance(model = model_treated)
importance_control <- xgb.importance(model = model_control)

cat("Top variables in treated model:\n")
print(head(importance_treated, 20))
cat("\n")

cat("Top variables in control model:\n")
print(head(importance_control, 20))
cat("\n")

# ============================================================
# 18. SCORE FULL FILE
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
    uplift_bad_outcome = pred_ed_if_treated - pred_ed_if_control
  )

scored_full$uplift_decile <- dplyr::ntile(dplyr::desc(scored_full$benefit_score), 10)

# ============================================================
# 19. WRITE OUTPUTS
# ============================================================

write_csv(scored_full, output_path)
write_csv(decile_summary, summary_path)

cat("Scored full file written to:\n", output_path, "\n\n")
cat("Decile summary written to:\n", summary_path, "\n\n")

# ============================================================
# 20. INTERPRETATION
# ============================================================

cat("INTERPRETATION:\n")
cat("- pred_ed_if_treated = predicted probability of ED within 90d if treated\n")
cat("- pred_ed_if_control = predicted probability of ED within 90d if not treated\n")
cat("- benefit_score = pred_ed_if_control - pred_ed_if_treated\n")
cat("- Higher benefit_score means treatment is predicted to reduce ED risk more\n")
cat("- Uplift decile 1 = highest predicted treatment benefit\n")

# ============================================================
# DASHBOARD VIEW
# ============================================================

library(ggplot2)
library(readr)
library(dplyr)

dashboard_folder <- "D:/Users/Rui.Huang/OneDrive - Acentra/Documents/PRISM"

# Chart 1: Average benefit by decile
p1 <- ggplot(decile_summary, aes(x = factor(uplift_decile), y = avg_benefit_score)) +
  geom_col() +
  labs(
    title = "Average Predicted Intervention Benefit by Uplift Decile",
    x = "Uplift Decile: 1 = Highest Predicted Benefit",
    y = "Average Benefit Score"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_avg_benefit_by_decile.png"),
  plot = p1,
  width = 8,
  height = 5
)

# Chart 2: Observed ED rate by decile
p2 <- ggplot(decile_summary, aes(x = factor(uplift_decile), y = observed_ed_rate)) +
  geom_col() +
  labs(
    title = "Observed 90-Day ED Rate by Uplift Decile",
    x = "Uplift Decile",
    y = "Observed ED Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_observed_ed_rate_by_decile.png"),
  plot = p2,
  width = 8,
  height = 5
)

# Chart 3: Treatment rate by decile
p3 <- ggplot(decile_summary, aes(x = factor(uplift_decile), y = treated_pct)) +
  geom_col() +
  labs(
    title = "Current Treatment Penetration by Uplift Decile",
    x = "Uplift Decile",
    y = "Percent Treated"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_treated_pct_by_decile.png"),
  plot = p3,
  width = 8,
  height = 5
)

# Chart 4: Predicted ED risk if treated vs control
decile_long <- decile_summary %>%
  select(uplift_decile, avg_pred_ed_if_treated, avg_pred_ed_if_control) %>%
  tidyr::pivot_longer(
    cols = c(avg_pred_ed_if_treated, avg_pred_ed_if_control),
    names_to = "scenario",
    values_to = "predicted_ed_rate"
  )

p4 <- ggplot(decile_long, aes(x = factor(uplift_decile), y = predicted_ed_rate, fill = scenario)) +
  geom_col(position = "dodge") +
  labs(
    title = "Predicted ED Risk: Treated vs Control by Decile",
    x = "Uplift Decile",
    y = "Predicted ED Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_predicted_treated_vs_control.png"),
  plot = p4,
  width = 9,
  height = 5
)

cat("Dashboard charts saved to:\n", dashboard_folder, "\n")

# ============================================================
# ROI PER DECILE
# ============================================================

# Assumptions — update these for your business case
cost_per_ed_visit <- 1200
cost_per_intervention <- 250

roi_summary <- decile_summary %>%
  mutate(
    expected_ed_rate_reduction = avg_benefit_score,
    expected_ed_visits_avoided = n * expected_ed_rate_reduction,
    gross_savings = expected_ed_visits_avoided * cost_per_ed_visit,
    intervention_cost = n * cost_per_intervention,
    net_savings = gross_savings - intervention_cost,
    roi = net_savings / intervention_cost
  )

print(roi_summary)

write_csv(
  roi_summary,
  file.path(dashboard_folder, "uplift_roi_by_decile.csv")
)

# ROI chart
p_roi <- ggplot(roi_summary, aes(x = factor(uplift_decile), y = net_savings)) +
  geom_col() +
  labs(
    title = "Estimated Net Savings by Uplift Decile",
    x = "Uplift Decile",
    y = "Estimated Net Savings"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_roi_net_savings_by_decile.png"),
  plot = p_roi,
  width = 8,
  height = 5
)

cat("ROI summary saved.\n")

# ============================================================
# SHAP EXPLANATIONS
# ============================================================

# xgboost has built-in SHAP contribution support
# This produces SHAP-style feature contribution values.

# For test set
shap_treated <- predict(
  model_treated,
  dtest,
  predcontrib = TRUE
)

shap_control <- predict(
  model_control,
  dtest,
  predcontrib = TRUE
)

# Convert to data frames
shap_treated_df <- as.data.frame(shap_treated)
shap_control_df <- as.data.frame(shap_control)

# Remove BIAS term for summary
shap_treated_no_bias <- shap_treated_df[, names(shap_treated_df) != "BIAS", drop = FALSE]
shap_control_no_bias <- shap_control_df[, names(shap_control_df) != "BIAS", drop = FALSE]

# Mean absolute SHAP values = global importance
shap_treated_importance <- data.frame(
  feature = names(shap_treated_no_bias),
  mean_abs_shap = sapply(shap_treated_no_bias, function(x) mean(abs(x), na.rm = TRUE)),
  model = "Treated Model"
) %>%
  arrange(desc(mean_abs_shap))

shap_control_importance <- data.frame(
  feature = names(shap_control_no_bias),
  mean_abs_shap = sapply(shap_control_no_bias, function(x) mean(abs(x), na.rm = TRUE)),
  model = "Control Model"
) %>%
  arrange(desc(mean_abs_shap))

shap_importance_combined <- bind_rows(
  shap_treated_importance,
  shap_control_importance
)

write_csv(
  shap_importance_combined,
  file.path(dashboard_folder, "shap_importance_treated_control_models.csv")
)

# Plot top 20 SHAP features for treated model
top_treated <- shap_treated_importance %>% slice_max(mean_abs_shap, n = 20)

p_shap_treated <- ggplot(top_treated, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top SHAP Drivers: Treated Model",
    x = "Feature",
    y = "Mean Absolute SHAP Contribution"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_shap_treated_model.png"),
  plot = p_shap_treated,
  width = 9,
  height = 6
)

# Plot top 20 SHAP features for control model
top_control <- shap_control_importance %>% slice_max(mean_abs_shap, n = 20)

p_shap_control <- ggplot(top_control, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top SHAP Drivers: Control Model",
    x = "Feature",
    y = "Mean Absolute SHAP Contribution"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(dashboard_folder, "dashboard_shap_control_model.png"),
  plot = p_shap_control,
  width = 9,
  height = 6
)

cat("SHAP outputs saved.\n")