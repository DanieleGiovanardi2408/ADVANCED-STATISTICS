# src/r/04b_calibrate_threshold_reg.R
library(readr); library(pROC); library(jsonlite)

# carica modello reg
fit_reg <- readRDS("outputs/model_artifacts/logit_reg_best.rds")
val_probs <- read_csv("outputs/val_probs_reg.csv", show_col_types = FALSE)

# carica test completo (o costruiscilo al volo)
test <- read_csv("data/interim/test_with_segment.csv", show_col_types = FALSE)
features <- c("SEX","EDUCATION","MARRIAGE","AGE",
              paste0("PAY_", c(0,2,3,4,5,6)),
              paste0("BILL_AMT",1:6),
              paste0("PAY_AMT",1:6),
              "max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend","segment")
X_te <- model.matrix(reformulate(features), data=test)[, -1]

# calibrazione (Platt) su validation del regolarizzato
platt_reg <- glm(default ~ prob_reg, data=val_probs, family="binomial")
saveRDS(platt_reg, "outputs/model_artifacts/platt_scaling_reg.rds")

test$prob_reg_raw <- as.numeric(predict(fit_reg, newx=X_te, type="response"))
test$pd_cal_reg   <- predict(platt_reg,
                             newdata=data.frame(prob_reg=test$prob_reg_raw),
                             type="response")

roc_val <- roc(val_probs$default,
               predict(platt_reg, newdata=val_probs, type="response"))
t_star  <- as.numeric(coords(roc_val, "best", best.method="youden")["threshold"])

pred <- as.integer(test$pd_cal_reg >= t_star)
TP <- sum(pred==1 & test$default==1)
FP <- sum(pred==1 & test$default==0)
TN <- sum(pred==0 & test$default==0)
FN <- sum(pred==0 & test$default==1)
auc_test <- as.numeric(roc(test$default, test$pd_cal_reg)$auc)

write_json(list(AUC_test=auc_test, threshold=t_star, TP=TP, FP=FP, TN=TN, FN=FN),
           "outputs/metrics_test_reg.json", pretty=TRUE)
cat(sprintf("REG TEST — AUC=%.3f | t*=%.2f | TP=%d FP=%d TN=%d FN=%d\n",
            auc_test, t_star, TP, FP, TN, FN))
