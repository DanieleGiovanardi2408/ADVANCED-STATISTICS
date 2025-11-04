install.packages(c("readxl","dplyr","caret","pROC","glmnet","car","jsonlite"))
path_src <- "/Users/danielegiovanardi/Desktop/ADVANCED STATISTICS/default of credit card clients.xls"
stopifnot(file.exists(path_src))
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
file.copy(path_src, "data/raw/default of credit card clients.xls", overwrite = TRUE)

set.seed(42)
library(readxl); library(dplyr); library(caret)

df <- read_excel("data/raw/default of credit card clients.xls", skip=1)
names(df)[names(df)=="default payment next month"] <- "default"
df <- df %>% mutate(row_id = dplyr::row_number()) %>% select(-ID)

# Ricodifica education/marriage out-of-range -> Other/Unknown
df <- df %>%
  mutate(
    EDUCATION = case_when(EDUCATION %in% c(1,2,3,4) ~ EDUCATION,
                          EDUCATION %in% c(0,5,6) ~ 4L, TRUE ~ 4L),
    MARRIAGE  = case_when(MARRIAGE %in% c(1,2,3) ~ MARRIAGE,
                          MARRIAGE %in% c(0) ~ 3L, TRUE ~ 3L),
    default = as.integer(default)
  )

# Split stratificato 60/20/20
idx_tr <- createDataPartition(df$default, p=0.6, list=FALSE)
train <- df[idx_tr, ]; rem <- df[-idx_tr, ]
idx_va <- createDataPartition(rem$default, p=0.5, list=FALSE)
val <- rem[idx_va, ]; test <- rem[-idx_va, ]

# crea le cartelle se non esistono
dir.create("data", showWarnings = FALSE)
dir.create("data/interim", recursive = TRUE, showWarnings = FALSE)

write.csv(train,"data/interim/train.csv",row.names=FALSE)
write.csv(val,  "data/interim/val.csv",  row.names=FALSE)
write.csv(test, "data/interim/test.csv", row.names=FALSE)
cat("OK: train/val/test salvati in data/interim/\n")

