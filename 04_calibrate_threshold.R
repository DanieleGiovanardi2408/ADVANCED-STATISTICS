# src/r/04_calibrate_threshold.R
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(pROC); library(jsonlite)
})

cat("WD:", getwd(), "\n")

# -------- helper: carica uno split con il segmento --------
load_split_with_segment <- function(split = c("train","val","test")) {
  split <- match.arg(split)
  path_ws <- file.path("data/interim", paste0(split, "_with_segment.csv"))
  if (file.exists(path_ws)) {
    cat(">> Loading", path_ws, "\n")
    df <- read_csv(path_ws, show_col_types = FALSE)
  } else {
    # costruiscilo al volo da *_features + segments_*
    path_feat <- file.path("data/interim", paste0(split, "_features.csv"))
    path_seg  <- file.path("data/interim", paste0("segments_", split, ".csv"))
    if (!file.exists(path_feat) || !file.exists(path_seg)) {
      stop("Manca uno dei file: ", path_feat, " oppure ", path_seg)
    }
    cat(">> Building join on the fly for", split, "\n")
    df <- read_csv(path_feat, show_col_types = FALSE) %>%
      left_join(read_csv(path_seg, show_col_types = FALSE), by = "row_id")
    if (any(is.na(df$segment))) stop("Segment NA found in ", split)
    # salviamo per usi futuri
    write_csv(df, path_ws)
    cat(">> Saved ", path_ws, "\n")
  }
  # assicuriamoci che 'segment' sia factor (come nel fit)
  df$segment <- factor(df$segment)
  df
}

# -------- 1) Carica oggetti necessari --------
# modello logit (baseline + segment) salvato nello step 03
fit1 <- readRDS("outputs/model_artifacts/logit_baseline.rds")

# predizioni di validation (prob1 del modello +segment)
val_probs <- read_csv("outputs/val_probs.csv", show_col_types = FALSE)

# dataset TEST completo (feature + segment)
test <- load_split_with_segment("test")

# -------- 2) Calibrazione (Platt) su validation --------
platt <- glm(default ~ prob1, data = val_probs, family = binomial())
dir.create("outputs/model_artifacts", recursive = TRUE, showWarnings = FALSE)
saveRDS(platt, "outputs/model_artifacts/platt_scaling.rds")

# -------- 3) Predizioni su test --------
# prob grezza dal modello logit su test
test$prob1_raw <- predict(fit1, newdata = test, type = "response")

# applica Platt -> PD calibrata
test$pd_cal <- predict(platt, newdata = data.frame(prob1 = test$prob1_raw), type = "response")

# -------- 4) Soglia da validation calibrata + metriche su test --------
roc_val <- roc(val_probs$default,
               predict(platt, newdata = val_probs, type = "response"))
t_star  <- as.numeric(coords(roc_val, "best", best.method = "youden")["threshold"])

pred_test <- as.integer(test$pd_cal >= t_star)
TP <- sum(pred_test==1 & test$default==1)
FP <- sum(pred_test==1 & test$default==0)
TN <- sum(pred_test==0 & test$default==0)
FN <- sum(pred_test==0 & test$default==1)
auc_test <- as.numeric(roc(test$default, test$pd_cal)$auc)

dir.create("outputs", showWarnings = FALSE)
write_json(list(AUC_test = auc_test,
                threshold = t_star, TP = TP, FP = FP, TN = TN, FN = FN),
           "outputs/metrics_test.json", pretty = TRUE)

dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)
png("reports/figures/roc_test.png", 800, 500)
plot(roc(test$default, test$pd_cal),
     col = "#1f77b4", main = sprintf("ROC (test) — AUC=%.3f", auc_test))
abline(0,1,col="grey80", lty=2)
dev.off()

cat(sprintf("TEST — AUC=%.3f | t*=%.2f | TP=%d FP=%d TN=%d FN=%d\n",
            auc_test, t_star, TP, FP, TN, FN))

# dopo il fit di fit1 (baseline + segment)
summ <- summary(fit1)$coefficients
coef_tbl <- data.frame(term = rownames(summ),
                       estimate = summ[, "Estimate"],
                       std.error = summ[, "Std. Error"],
                       z = summ[, "z value"],
                       p.value = summ[, "Pr(>|z|)"])
readr::write_csv(coef_tbl, "outputs/logit_coefficients_pvalues.csv")

cm <- matrix(c(TP, FP, FN, TN), nrow=2, byrow=TRUE,
             dimnames=list("Pred"=c("Default","NoDefault"),
                           "True"=c("Default","NoDefault")))
metrics <- list(
  AUC_test = auc_test,
  threshold = t_star,
  TP=TP, FP=FP, TN=TN, FN=FN,
  accuracy = (TP+TN)/ (TP+FP+TN+FN),
  precision = TP/(TP+FP),
  recall = TP/(TP+FN),
  specificity = TN/(TN+FP),
  F1 = 2*(TP/(TP+FP))*(TP/(TP+FN)) / ((TP/(TP+FP))+(TP/(TP+FN))),
  balanced_accuracy = ((TP/(TP+FN)) + (TN/(TN+FP)))/2
)
jsonlite::write_json(metrics, "outputs/metrics_test_detailed.json", pretty=TRUE)
write.csv(cm, "outputs/confusion_matrix_test.csv")

