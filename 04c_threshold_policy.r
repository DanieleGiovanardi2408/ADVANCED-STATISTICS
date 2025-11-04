# src/r/04c_threshold_policy.R
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(pROC); library(jsonlite); library(ggplot2)
})

# --- 0) Parametri business da ADATTARE ---
# Opzione semplice: costo relativo FN vs FP (FN = accettare un default)
cost_FP <- 1
cost_FN <- 5
# Opzione credit: se hai EAD/LGD per-cliente, puoi sostituire con colonne test$EAD * test$LGD ecc.

# Opzione C: vincolo tasso di accettazione (es. approvo il 70%)
target_approval <- NA_real_  # metti 0.70 per attivare, oppure lascia NA per ignorare

# --- 1) Carica oggetti del modello baseline + segment (già esistenti) ---
fit1  <- readRDS("outputs/model_artifacts/logit_baseline.rds")
platt <- readRDS("outputs/model_artifacts/platt_scaling.rds")

# Validation: row_id, default, prob1 già salvati da 03_logit_baseline.R
val_probs <- read_csv("outputs/val_probs.csv", show_col_types = FALSE)
val_probs$pd_cal <- predict(platt, newdata = val_probs, type = "response")

# Test completo (costruiscilo al volo se manca _with_segment)
load_split_with_segment <- function(split) {
  path_ws <- file.path("data/interim", paste0(split, "_with_segment.csv"))
  if (!file.exists(path_ws)) {
    feats <- read_csv(file.path("data/interim", paste0(split, "_features.csv")),
                      show_col_types = FALSE)
    segs  <- read_csv(file.path("data/interim", paste0("segments_", split, ".csv")),
                      show_col_types = FALSE)
    df <- left_join(feats, segs, by = "row_id")
    write_csv(df, path_ws)
  }
  df <- read_csv(path_ws, show_col_types = FALSE)
  df$segment <- factor(df$segment)
  df
}
test <- load_split_with_segment("test")
test$prob1_raw <- predict(fit1, newdata=test, type="response")
test$pd_cal    <- predict(platt, newdata=data.frame(prob1=test$prob1_raw), type="response")

# --- 2) Soglia A: Youden (di riferimento) su validation calibrata ---
roc_val <- roc(val_probs$default, val_probs$pd_cal)
t_youden <- as.numeric(coords(roc_val, "best", best.method="youden")["threshold"])

# --- 3) Soglia B: costo-minimo su validation ---
grid_t <- seq(0.02, 0.60, by=0.005)  # esplora soglie ragionevoli
cost_fun <- function(t) {
  pred <- as.integer(val_probs$pd_cal >= t)
  TP <- sum(pred==1 & val_probs$default==1)
  FP <- sum(pred==1 & val_probs$default==0)
  TN <- sum(pred==0 & val_probs$default==0)
  FN <- sum(pred==0 & val_probs$default==1)
  # costo = FP*cost_FP + FN*cost_FN (normalizza per n per confronti)
  (FP * cost_FP + FN * cost_FN) / nrow(val_probs)
}
costs <- sapply(grid_t, cost_fun)
t_cost <- grid_t[ which.min(costs) ]

# --- 4) Soglia C: tasso di accettazione target (approvo se PD < t) ---
t_approval <- NA_real_
if (!is.na(target_approval)) {
  t_approval <- unname( quantile(val_probs$pd_cal, probs = target_approval, type = 7) )
}

# --- 5) Scegli quale soglia usare: costo-minimo (default) ---
t_star <- t_cost
policy  <- sprintf("cost-min (FN=%g, FP=%g)", cost_FN, cost_FP)

# Se vuoi usare Youden:
# t_star <- t_youden; policy <- "youden"

# Se vuoi usare approval rate:
# if (!is.na(t_approval)) { t_star <- t_approval; policy <- sprintf("approval=%.0f%%", 100*target_approval) }

# --- 6) Metriche su TEST con t_star ---
pred_test <- as.integer(test$pd_cal >= t_star)
TP <- sum(pred_test==1 & test$default==1)
FP <- sum(pred_test==1 & test$default==0)
TN <- sum(pred_test==0 & test$default==0)
FN <- sum(pred_test==0 & test$default==1)
auc_test <- as.numeric(roc(test$default, test$pd_cal)$auc)

acc  <- (TP+TN)/nrow(test)
prec <- ifelse(TP+FP>0, TP/(TP+FP), NA)
rec  <- TP/(TP+FN)
spec <- TN/(TN+FP)
F1   <- ifelse(is.na(prec), NA, 2*prec*rec/(prec+rec))
bal  <- (rec + spec)/2

dir.create("outputs", showWarnings=FALSE)
out <- list(
  policy = policy, threshold = t_star,
  AUC_test = auc_test, accuracy = acc, precision = prec, recall = rec,
  specificity = spec, F1 = F1, balanced_accuracy = bal,
  TP=TP, FP=FP, TN=TN, FN=FN,
  threshold_youden = t_youden,
  threshold_costmin = t_cost,
  threshold_approval = t_approval
)
jsonlite::write_json(out, "outputs/metrics_test_policy.json", pretty=TRUE)

# --- 7) Grafici utili ---
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

# ROC test con t_star
png("reports/figures/roc_test_policy.png", 800, 500)
plot(roc(test$default, test$pd_cal), col="#1f77b4",
     main=sprintf("ROC (test) — AUC=%.3f | policy=%s | t*=%.2f", auc_test, policy, t_star))
abline(0,1,col="grey80", lty=2)
dev.off()

# Curva costo vs soglia (validation)
dfc <- data.frame(t = grid_t, cost = costs)
png("reports/figures/cost_curve_validation.png", 800, 500)
ggplot(dfc, aes(t, cost)) + geom_line() + geom_vline(xintercept=t_cost, linetype=2, color="red") +
  labs(title="Validation cost curve (cost-min threshold in red)",
       x="threshold (PD)", y="normalized cost (FP*cfp + FN*cfn)/N")
dev.off()

cat(sprintf("POLICY: %s | t*=%.3f | TEST: AUC=%.3f Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f\n",
            policy, t_star, auc_test, acc, prec, rec, F1))
