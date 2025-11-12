# scripts/z_io_helpers.R
suppressPackageStartupMessages({ library(readr); library(dplyr) })

find_target <- function(nms) intersect(c("default","y","target","label"), nms)[1]

read_probs <- function(split = c("train","val","test"), model = c("logit","rf")) {
  split <- match.arg(split); model <- match.arg(model)
  base <- file.path("data/interim", sprintf("%s_with_probs%s.csv",
                                            split, ifelse(model=="rf","_rf","")))
  if (!file.exists(base)) stop("Manca: ", base)
  df <- suppressMessages(readr::read_csv(base, show_col_types = FALSE))
  ycol <- find_target(names(df)); if (is.na(ycol)) stop("Target non trovato in ", base)
  if (model=="logit") {
    pcol <- intersect(c("prob","prob_logit","probability","pred_prob","p","yhat","score",".fitted","pred","prob1"),
                      names(df))[1]
  } else {
    pcol <- intersect(c("prob_rf","rf_prob","rf_score"), names(df))[1]
  }
  if (is.na(pcol)) stop("Colonna probabilità non trovata in ", base)
  tibble(y = as.integer(df[[ycol]]),
         prob = pmin(pmax(as.numeric(df[[pcol]]), 1e-8), 1-1e-8))
}
