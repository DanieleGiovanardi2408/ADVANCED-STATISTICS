library(glmnet); library(pROC); library(readr)

train <- read_csv("data/interim/train_with_segment.csv", show_col_types=FALSE)
val   <- read_csv("data/interim/val_with_segment.csv",   show_col_types=FALSE)

feat <- c("SEX","EDUCATION","MARRIAGE","AGE",
          paste0("PAY_", c(0,2,3,4,5,6)),
          paste0("BILL_AMT",1:6), paste0("PAY_AMT",1:6),
          "max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend","segment")

Xtr <- model.matrix(reformulate(feat), data=train)[,-1]
ytr <- train$default
Xva <- model.matrix(reformulate(feat), data=val)[,-1]
yva <- val$default

alphas <- c(0, .25, .5, .75, 1)
best <- NULL
for (a in alphas) {
  set.seed(42)
  cv <- cv.glmnet(Xtr, ytr, family="binomial", alpha=a, nfolds=5,
                  type.measure="auc")  # <— qui
  for (lam in c(cv$lambda.min, cv$lambda.1se)) {
    fit <- glmnet(Xtr, ytr, family="binomial", alpha=a, lambda=lam)
    pva <- as.numeric(predict(fit, newx=Xva, type="response"))
    auc <- as.numeric(pROC::roc(yva, pva)$auc)
    cat(sprintf("alpha=%.2f | λ=%.4g | AUC(val)=%.3f\n", a, lam, auc))
    if (is.null(best) || auc > best$auc) best <- list(alpha=a, lambda=lam, fit=fit, auc=auc)
  }
}
saveRDS(best$fit, "outputs/model_artifacts/logit_reg_best_auc.rds")
writeLines(sprintf("best_alpha=%.2f\nbest_lambda=%.6g\nauc_val=%.3f",
                   best$alpha, best$lambda, best$auc),
           "outputs/logit_reg_auc_summary.txt")
