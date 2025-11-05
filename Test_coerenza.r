library(readr); library(dplyr)

val_probs <- read_csv("outputs/val_probs.csv", show_col_types=FALSE)
platt     <- readRDS("outputs/model_artifacts/platt_scaling.rds")
val_probs$pd_cal <- predict(platt, newdata=val_probs, type="response")

test <- read_csv("data/interim/test_with_segment.csv", show_col_types=FALSE)
fit1 <- readRDS("outputs/model_artifacts/logit_baseline.rds")
test$pd_cal <- predict(platt,
                       newdata=data.frame(prob1=predict(fit1, newdata=test, type="response")),
                       type="response")

# dimensioni e prevalenza
sapply(list(val=val_probs, test=test), \(x) c(n=nrow(x), prev=mean(x$default)))
# PD in [0,1] e distribuzione
summary(val_probs$pd_cal); summary(test$pd_cal)


library(pROC)
roc_val <- roc(val_probs$default, val_probs$pd_cal)
auc(roc_val)              # ~0.75
coords(roc_val, "best", best.method="youden")  # soglia ~0.20

roc_test <- roc(test$default, test$pd_cal)
auc(roc_test)             # ~0.759


cost_FP <- 1; cost_FN <- 5
grid_t <- seq(0.02, 0.6, by=0.005)
costs <- sapply(grid_t, function(t){
  pred <- as.integer(val_probs$pd_cal >= t)
  FP <- sum(pred==1 & val_probs$default==0)
  FN <- sum(pred==0 & val_probs$default==1)
  (FP*cost_FP + FN*cost_FN)/nrow(val_probs)
})
grid_t[which.min(costs)]



t_cost <- grid_t[which.min(costs)]
c(
  approval_rate_val  = mean(val_probs$pd_cal < t_cost),
  approval_rate_test = mean(test$pd_cal < t_cost)
)



youden <- as.numeric(coords(roc_val, "best", best.method="youden")["threshold"])
eval_at <- function(t, df) {
  pred <- as.integer(df$pd_cal >= t)
  TP <- sum(pred==1 & df$default==1); FP <- sum(pred==1 & df$default==0)
  TN <- sum(pred==0 & df$default==0); FN <- sum(pred==0 & df$default==1)
  c(t=t, Prec=TP/(TP+FP), Rec=TP/(TP+FN), Spec=TN/(TN+FP),
    Acc=(TP+TN)/nrow(df), F1=2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))),
    TP=TP, FP=FP, TN=TN, FN=FN)
}
as.data.frame(rbind(
  Youden   = eval_at(youden, test),
  Cost_min = eval_at(t_cost, test)
))

