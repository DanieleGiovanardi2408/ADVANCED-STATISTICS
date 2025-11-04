
library(dplyr); library(tidyr)

mk_features <- function(df){
  pay  <- c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")
  bill <- paste0("BILL_AMT",1:6)
  pamt <- paste0("PAY_AMT",1:6)
  df %>%
    mutate(
      max_dpd  = pmax(!!!syms(pay), na.rm=TRUE),
      cnt_dpd  = rowSums(across(all_of(pay), ~ .x > 0), na.rm=TRUE),
      util_last= BILL_AMT1 / ifelse(LIMIT_BAL==0, NA, LIMIT_BAL),
      util_last= pmin(pmax(util_last,0),5),
      pay_ratio= rowSums(across(all_of(pamt)), na.rm=TRUE) /
                 ifelse(rowSums(across(all_of(bill)), na.rm=TRUE)==0, NA,
                        rowSums(across(all_of(bill)), na.rm=TRUE)),
      pay_ratio= pmin(pmax(replace_na(pay_ratio,0),0),3),
      bill_trend = { t <- 1:6
        apply(select(., all_of(bill)), 1, function(v){ v[is.na(v)]<-0; coef(lm(v~t))[2] })
      }
    ) %>% mutate(across(c(util_last,pay_ratio,bill_trend), ~replace_na(.,0)))
}

train <- read.csv("data/interim/train.csv")
val   <- read.csv("data/interim/val.csv")
test  <- read.csv("data/interim/test.csv")

write.csv(mk_features(train), "data/interim/train_features.csv", row.names=FALSE)
write.csv(mk_features(val),   "data/interim/val_features.csv",   row.names=FALSE)
write.csv(mk_features(test),  "data/interim/test_features.csv",  row.names=FALSE)
cat("OK: feature files in data/interim/\n")

