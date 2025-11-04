library(pROC)

pay  <- c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")
bill <- paste0("BILL_AMT",1:6)
pamt <- paste0("PAY_AMT",1:6)
eng  <- c("max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend")
form_base <- as.formula(paste(
  "default ~", paste(c("SEX","EDUCATION","MARRIAGE","AGE", pay, bill, pamt, eng), collapse="+")
))

fit0 <- glm(form_base, data=train, family=binomial())
fit1 <- glm(update(form_base, . ~ . + factor(segment)), data=train, family=binomial())

val$prob0 <- predict(fit0, val, type="response")
val$prob1 <- predict(fit1, val, type="response")

auc0 <- as.numeric(roc(val$default, val$prob0)$auc)
auc1 <- as.numeric(roc(val$default, val$prob1)$auc)
cat("Validation AUC — base:", round(auc0,3), " | +segment:", round(auc1,3), "\n")

# === SAVE ROC + METRICHE (VALIDATION) ===
library(pROC); library(ggplot2); library(jsonlite)

roc0 <- roc(val$default, val$prob0)
roc1 <- roc(val$default, val$prob1)

png("reports/figures/roc_validation.png", 800, 500)
plot(roc0, col="#999999", main=sprintf("ROC (val) — base=%.3f, +seg=%.3f", auc0, auc1))
plot(roc1, col="#1f77b4", add=TRUE); legend("bottomright",
                                            legend=c(sprintf("base AUC=%.3f",auc0), sprintf("+seg AUC=%.3f",auc1)),
                                            col=c("#999999","#1f77b4"), lwd=2, bty="n")
dev.off()

write_json(list(auc_base=auc0, auc_plus_segment=auc1),
           "outputs/validation_metrics.json", pretty=TRUE)

# (comodo per lo step successivo)
dir.create("outputs/model_artifacts", recursive=TRUE, showWarnings=FALSE)
saveRDS(fit1, "outputs/model_artifacts/logit_baseline.rds")
readr::write_csv(val[, c("row_id","default","prob1")], "outputs/val_probs.csv")




