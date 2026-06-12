library(tidyverse)
library(gt)

roadmap <- tribble(
  ~Section, ~Task, ~May, ~Jun, ~Jul, ~Aug, ~Sep, ~Oct, ~Nov, ~Dec,

  "Program Governance & Strategy", "Project Charter & Governance", "█", "█", "█", "", "", "", "", "",
  "Program Governance & Strategy", "Executive Sponsorship Alignment", "█", "█", "█", "", "", "", "", "",
  "Program Governance & Strategy", "Success Metrics / KPI Definition", "█", "█", "█", "", "", "", "", "",
  "Program Governance & Strategy", "Steering Committee & Reporting", "█", "█", "█", "█", "█", "█", "█", "█",

  "Workstream A — Data & Feature Engineering", "Enterprise Data Inventory", "█", "█", "█", "", "", "", "", "",
  "Workstream A — Data & Feature Engineering", "Canonical Member Timeline", "█", "█", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "Historical Intervention Mapping", "█", "█", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "Feature Store Design", "█", "█", "█", "█", "", "", "", "",
  "Workstream A — Data & Feature Engineering", "Clinical Feature Engineering", "", "█", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "SDOH Feature Engineering", "", "█", "█", "█", "█", "█", "", "",
  "Workstream A — Data & Feature Engineering", "Behavioral / Engagement Features", "", "█", "█", "█", "█", "█", "", "",
  "Workstream A — Data & Feature Engineering", "Temporal Feature Engineering", "", "█", "█", "█", "█", "█", "", "",
  "Workstream A — Data & Feature Engineering", "Data Quality & Validation", "█", "█", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "Reusable Pipeline Development", "█", "█", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "Feature Lineage Documentation", "", "", "█", "█", "█", "", "", "",
  "Workstream A — Data & Feature Engineering", "MILESTONE: FEATURE STORE V1", "", "", "", "", "▓", "", "", "",

  "Workstream B — Baseline Predictive Modeling", "Outcome Variable Prioritization", "█", "█", "█", "", "", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "ED Utilization Model Development", "", "█", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Admission Risk Model Development", "", "█", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "High-Cost Escalation Model", "", "█", "█", "█", "", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Rising Risk Identification Model", "", "█", "█", "█", "", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Validation Framework", "", "█", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Calibration & Threshold Testing", "", "", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Performance Benchmarking", "", "", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "Explainability Layer (SHAP)", "", "", "█", "█", "█", "", "", "",
  "Workstream B — Baseline Predictive Modeling", "MILESTONE: CORE MODELS VALIDATED", "", "", "", "", "▓", "", "", "",

  "Workstream C — Automation & MLOps", "Automation Architecture Design", "", "█", "█", "", "", "", "", "",
  "Workstream C — Automation & MLOps", "Automated Scoring Pipelines", "", "", "█", "█", "█", "", "", "",
  "Workstream C — Automation & MLOps", "Model Retraining Pipelines", "", "", "█", "█", "█", "", "", "",
  "Workstream C — Automation & MLOps", "Monitoring & Drift Detection", "", "", "█", "█", "█", "", "", "",
  "Workstream C — Automation & MLOps", "Version Control & Auditability", "", "", "█", "█", "█", "", "", "",
  "Workstream C — Automation & MLOps", "Production Scheduling", "", "", "", "█", "█", "", "", "",
  "Workstream C — Automation & MLOps", "MILESTONE: AUTOMATED MODEL OPERATIONS", "", "", "", "", "▓", "", "", "",

  "Workstream D — UI / Productization (LENS)", "LENS Requirements & UX Design", "█", "█", "█", "", "", "", "", "",
  "Workstream D — UI / Productization (LENS)", "Wireframes & Dashboard Design", "", "█", "█", "█", "", "", "", "",
  "Workstream D — UI / Productization (LENS)", "Risk Prioritization Dashboard", "", "", "█", "█", "█", "", "", "",
  "Workstream D — UI / Productization (LENS)", "Explainability Dashboard", "", "", "█", "█", "█", "", "", "",
  "Workstream D — UI / Productization (LENS)", "Simulation Dashboard", "", "", "", "█", "█", "█", "", "",
  "Workstream D — UI / Productization (LENS)", "Subgroup Explorer", "", "", "", "", "", "█", "█", "█",
  "Workstream D — UI / Productization (LENS)", "UAT & Stakeholder Feedback", "", "", "█", "█", "█", "", "", "",
  "Workstream D — UI / Productization (LENS)", "MILESTONE: PRISM LENS RELEASE V1", "", "", "", "", "▓", "", "", "",
  "Workstream D — UI / Productization (LENS)", "MILESTONE: ADVANCED PRISM UI", "", "", "", "", "", "", "", "▓",

  "Workstream E — Governance & Compliance", "Model Governance Framework", "█", "█", "█", "█", "", "", "", "",
  "Workstream E — Governance & Compliance", "Audit & Reproducibility Standards", "", "█", "█", "█", "█", "", "", "",
  "Workstream E — Governance & Compliance", "Bias / Equity Review", "", "█", "█", "█", "█", "", "", "",
  "Workstream E — Governance & Compliance", "Clinical SME Validation", "", "█", "█", "█", "█", "", "", "",
  "Workstream E — Governance & Compliance", "Documentation & SOP Development", "", "", "█", "█", "█", "█", "", "",
  "Workstream E — Governance & Compliance", "MILESTONE: GOVERNANCE PACKAGE COMPLETE", "", "", "", "", "", "", "", "▓",

  "Phase 1 — September 2026 Delivery", "GO-LIVE: SEPTEMBER 2026", "", "", "", "", "▓", "", "", "",

  "Workstream F — Advanced Causal & Uplift Modeling", "Treatment / Control Cohorts", "", "", "", "", "", "█", "█", "",
  "Workstream F — Advanced Causal & Uplift Modeling", "Uplift Modeling Development", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Meta-Learners (T/X/DR)", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Counterfactual Estimation", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Individual Treatment Effects", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Causal Forests / HTE Modeling", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Subgroup Optimization", "", "", "", "", "", "", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "Sensitivity & Robustness Testing", "", "", "", "", "", "█", "█", "█",
  "Workstream F — Advanced Causal & Uplift Modeling", "MILESTONE: ADVANCED CAUSAL MODELS", "", "", "", "", "", "", "", "▓",

  "Workstream G — Simulation Engine", "Simulation Framework Design", "", "", "", "", "", "█", "█", "",
  "Workstream G — Simulation Engine", "Alternative Strategy Engine", "", "", "", "", "", "█", "█", "█",
  "Workstream G — Simulation Engine", "ROI / Cost Avoidance Logic", "", "", "", "", "", "█", "█", "█",
  "Workstream G — Simulation Engine", "Policy Simulation Modeling", "", "", "", "", "", "█", "█", "█",
  "Workstream G — Simulation Engine", "Scenario Optimization", "", "", "", "", "", "█", "█", "█",
  "Workstream G — Simulation Engine", "Strategy Comparison Framework", "", "", "", "", "", "█", "█", "█",
  "Workstream G — Simulation Engine", "MILESTONE: SIMULATION ENGINE LIVE", "", "", "", "", "", "", "", "▓",

  "Phase 2 — December 2026 Delivery", "GO-LIVE: DECEMBER 2026", "", "", "", "", "", "", "", "▓",

  "Outcome Model Delivery Schedule — Foundational Models", "90-Day ED Prediction Model", "█", "█", "█", "█", "▓", "", "", "",
  "Outcome Model Delivery Schedule — Foundational Models", "90-Day Admission Prediction Model", "█", "█", "█", "█", "▓", "", "", "",
  "Outcome Model Delivery Schedule — Foundational Models", "High-Cost Escalation Model", "█", "█", "█", "█", "▓", "", "", "",
  "Outcome Model Delivery Schedule — Foundational Models", "Rising Risk Identification", "█", "█", "█", "█", "▓", "", "", "",

  "Outcome Model Delivery Schedule — Advanced Models", "Readmission Prediction Model", "", "", "", "", "", "█", "█", "▓",
  "Outcome Model Delivery Schedule — Advanced Models", "Avoidable Admission Model", "", "", "", "", "", "█", "█", "▓",
  "Outcome Model Delivery Schedule — Advanced Models", "Medication Adherence Model", "", "", "", "", "", "█", "█", "▓",
  "Outcome Model Delivery Schedule — Advanced Models", "Quality Gap Closure Prediction", "", "", "", "", "", "█", "█", "▓",
  "Outcome Model Delivery Schedule — Advanced Models", "Behavioral Health Escalation", "", "", "", "", "", "█", "█", "▓",
  "Outcome Model Delivery Schedule — Advanced Models", "SDOH Instability Prediction", "", "", "", "", "", "█", "█", "▓"
)

roadmap_table <- roadmap %>%
  gt(groupname_col = "Section") %>%
  tab_header(
    title = md("**Project PRISM Roadmap**"),
    subtitle = "May–December 2026"
  ) %>%
  cols_label(
    Task = "Workstream / Phase"
  ) %>%
  cols_align(
    align = "left",
    columns = Task
  ) %>%
  cols_align(
    align = "center",
    columns = May:Dec
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_row_groups()
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      rows = str_detect(Task, "MILESTONE|GO-LIVE")
    )
  ) %>%
  tab_options(
    table.font.size = px(12),
    data_row.padding = px(4),
    row_group.padding = px(6),
    table.width = pct(100)
  ) %>%
  tab_source_note(
    source_note = md("**Legend:** █ = Active Work &nbsp;&nbsp; ▓ = Major Milestone / Go-Live &nbsp;&nbsp; ▒ = Stabilization / Optimization")
  )
gtsave(
  roadmap_table,
  filename = "D:/Users/Rui.Huang/OneDrive - Acentra/Documents/PRISM/PRISM_Roadmap.pdf"
)
roadmap_table
