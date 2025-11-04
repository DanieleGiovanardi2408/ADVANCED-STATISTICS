# src/r/03b_logit_regularized.R
library(glmnet); library(pROC); library(dplyr); library(readr)

# carica split con segment già unito (o fai il join come prima)
train <- read_csv("data/interim/train_with_segment.csv", show_col_types = FALSE)
val   <- read_csv("data/interim/val_with_segment.csv",   show_col_types = FALSE)

# formula di base (come prima) + segment
pay  <- c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")
bill <- paste0("BILL_AMT",1:6)
pamt <- paste0("PAY_AMT",1:6)
eng  <- c("max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend")
features <- c("SEX","EDUCATION","MARRIAGE","AGE", pay, bill, pamt, eng, "segment")

# model.matrix crea automaticamente le dummies (drop intercept [-1] se vuoi lasciare a glmnet)
X_tr <- model.matrix(reformulate(features), data=train)[, -1]
y_tr <- train$default
X_va <- model.matrix(reformulate(features), data=val)[, -1]
y_va <- val$default

alphas <- c(0, 0.25, 0.5, 0.75, 1)  # ridge..lasso
res <- list()

for (a in alphas) {
  set.seed(42)
  cv <- cv.glmnet(X_tr, y_tr, family="binomial", alpha=a, nfolds=5, type.measure="deviance")
  fit <- glmnet(X_tr, y_tr, family="binomial", alpha=a, lambda=cv$lambda.min)
  p_va <- as.numeric(predict(fit, newx=X_va, type="response"))
  auc  <- as.numeric(roc(y_va, p_va)$auc)
  res[[as.character(a)]] <- list(alpha=a, lambda=cv$lambda.min, auc=auc, fit=fit)
  cat(sprintf("alpha=%.2f | lambda=%.4g | AUC(val)=%.3f\n", a, cv$lambda.min, auc))
}

# scegli il migliore
best <- res[[which.max(sapply(res, `[[`, "auc"))]]
saveRDS(best$fit, "outputs/model_artifacts/logit_reg_best.rds")
writeLines(sprintf("best_alpha=%.2f\nbest_lambda=%.6g\nauc_val=%.3f",
                   best$alpha, best$lambda, best$auc),
           "outputs/logit_reg_summary.txt")

# salva le proba validation per calibrazione come prima
val$prob_reg <- as.numeric(predict(best$fit, newx=X_va, type="response"))
readr::write_csv(val[, c("row_id","default","prob_reg")], "outputs/val_probs_reg.csv")
