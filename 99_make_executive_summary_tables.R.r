# scripts/99_make_executive_summary_tables.R — one-click executive tables
suppressPackageStartupMessages({ library(readr); library(dplyr); library(stringr) })
dir.create("outputs/metrics", TRUE, TRUE)

safe_read <- function(p){ if (file.exists(p)) suppressMessages(read_csv(p, show_col_types = FALSE)) else NULL }

auc_ci   <- safe_read("outputs/metrics/auc_ci_test.csv")
delong   <- safe_read("outputs/metrics/delong_logit_vs_rf.csv")
brier    <- safe_read("outputs/metrics/brier_decomposition_summary.csv")
brier_t  <- safe_read("outputs/metrics/brier_decomposition_test.csv")
rfmt     <- safe_read("outputs/metrics/rf_val_test.csv")
ablation <- safe_read("outputs/metrics/ablation.csv")
hl_bins  <- safe_read("outputs/metrics/hl_bins.csv")
hl_test  <- if (file.exists("outputs/metrics/hl_test.txt"))
  readLines("outputs/metrics/hl_test.txt") else character(0)

T0 <- tibble(
  item = c("AUC Logit (test)", "CI95% low", "CI95% high",
           "AUC RF (test)", "ΔAUC (RF-Logit)", "DeLong p-value",
           "Brier Logit (test)", "Brier RF (test)",
           "HL Test (note)"),
  value = c(
    if (!is.null(auc_ci)) sprintf("%.3f", auc_ci$auc[1]) else NA,
    if (!is.null(auc_ci)) sprintf("%.3f", auc_ci$ci_low[1]) else NA,
    if (!is.null(auc_ci)) sprintf("%.3f", auc_ci$ci_high[1]) else NA,
    if (!is.null(rfmt))   sprintf("%.3f", rfmt$auc[rfmt$set=="test"]) else NA,
    if (!is.null(rfmt) && !is.null(auc_ci))
      sprintf("%.3f", rfmt$auc[rfmt$set=="test"] - auc_ci$auc[1]) else NA,
    if (!is.null(delong)) signif(delong$p_value[1],3) else NA,
    if (!is.null(brier_t)) sprintf("%.4f", brier_t$bs[1]) else NA,
    if (!is.null(rfmt))    sprintf("%.4f", rfmt$brier[rfmt$set=="test"]) else NA,
    if (length(hl_test))   str_trim(paste(hl_test, collapse=" ")) else "n/a"
  )
)
write_csv(T0, "outputs/metrics/T0_executive.csv")

if (!is.null(ablation)) write_csv(ablation, "outputs/metrics/T1_ablation.csv")
if (!is.null(brier))    write_csv(brier,    "outputs/metrics/T2_brier_summary.csv")
if (!is.null(rfmt))     write_csv(rfmt,     "outputs/metrics/T3_rf_val_test.csv")
if (!is.null(auc_ci))   write_csv(auc_ci,   "outputs/metrics/T4_auc_ci.csv")
if (!is.null(delong))   write_csv(delong,   "outputs/metrics/T5_delong.csv")
if (!is.null(hl_bins))  write_csv(hl_bins,  "outputs/metrics/T6_hl_bins.csv")

cat("Executive tables create in outputs/metrics/: T0 + (T1..T6 se disponibili)\n")
