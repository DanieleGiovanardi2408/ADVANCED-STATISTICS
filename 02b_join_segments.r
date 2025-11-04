library(dplyr); library(readr)

train <- read_csv("data/interim/train_features.csv") |>
  left_join(read_csv("data/interim/segments_train.csv"), by="row_id")
val <- read_csv("data/interim/val_features.csv") |>
  left_join(read_csv("data/interim/segments_val.csv"), by="row_id")
test <- read_csv("data/interim/test_features.csv") |>
  left_join(read_csv("data/interim/segments_test.csv"), by="row_id")

stopifnot(!any(is.na(train$segment)), !any(is.na(val$segment)), !any(is.na(test$segment)))


library(dplyr); library(readr)

read_csv("data/interim/train_features.csv", show_col_types=FALSE) |>
  left_join(read_csv("data/interim/segments_train.csv", show_col_types=FALSE), by="row_id") |>
  write_csv("data/interim/train_with_segment.csv")

read_csv("data/interim/val_features.csv", show_col_types=FALSE) |>
  left_join(read_csv("data/interim/segments_val.csv", show_col_types=FALSE), by="row_id") |>
  write_csv("data/interim/val_with_segment.csv")
