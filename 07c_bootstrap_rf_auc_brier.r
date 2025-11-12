# --- RF: bootstrap CI per AUC/Brier (raw + Platt + Isotonic) ---
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(pROC)
})

set.seed(7)
dir.create("outputs/metrics", TRUE, TRUE)
dir.create("outputs/figures", TRUE, TRUE)

pick <- function(...) { x <- c(...); x[file.exists(x)][1] }
find_target <- function(nms) intersect(c("default","y","target","label"), nms)[1]
brier <- function(p,y) mean((p - y)^2)

# ---- carica val/test (prob RF raw su entrambi; useremo val per calibrare)
val_path  <- pick("data/interim/val_with_probs_rf.csv")
test_path <- pick("data/interim/test_with_probs_rf.csv")
stopifnot(!is.na(val_path), !is.na(test_path))

val  <- suppressMessages(read_csv(val_path,  show_col_types = FALSE))
test <- suppressMessages(read_csv(test_path, show_col_types = FALSE))

ycol <- find_target(names(val)); stopifnot(!is.na(ycol))
stopifnot("prob_rf" %in% names(val), "prob_rf" %in% names(test))
y_val  <- as.integer(val[[ycol]])
y_test <- as.integer(test[[ycol]])
p_raw_val  <- pmin(pmax(val$prob_rf,  1e-6), 1-1e-6)
p_raw_test <- pmin(pmax(test$prob_rf, 1e-6), 1-1e-6)

# ---- Calibrazioni post-hoc addestrate su validation ----
# Platt (logit su logit link)
platt_fit <- glm(y_val ~ qlogis(p_raw_val), family=binomial())
p_platt_test <- predict(
  platt_fit,
  newdata = data.frame(`qlogis(p_raw_val)` = qlogis(p_raw_test)),
  type = "response"
)

# Isotonic (ORDINA x e usa i fitted 'yf' come stepfun)
ord <- order(p_raw_val)                       # <— fix: ordina x
iso <- stats::isoreg(p_raw_val[ord], y_val[ord])
# rimuovi eventuali duplicati in x per lo stepfun
xk  <- iso$x
yk  <- iso$yf
keep <- c(TRUE, diff(xk) > 0)
xk  <- xk[keep];  yk <- yk[keep]
iso_fun <- stats::stepfun(xk, c(yk[1], yk))   # F(x) a gradini
p_iso_test <- iso_fun(p_raw_test)
p_iso_test <- pmin(pmax(p_iso_test, 1e-6), 1-1e-6)

# ---- Funzioni bootstrap ----
boot_n <- 2000
boot_ci <- function(y, p, fun){
  B <- boot_n
  idx <- matrix(sample.int(length(y), length(y)*B, TRUE), nrow=B)
  qs <- sapply(1:B, function(i) fun(y[idx[i,]], p[idx[i,]]))
  stats::quantile(qs, c(.025,.5,.975), na.rm=TRUE)
}
auc_fun   <- function(y,p) as.numeric(pROC::auc(pROC::roc(y,p,quiet=TRUE)))
brier_fun <- function(y,p) brier(p,y)

# ---- Calcolo metriche & CI ----
models <- list(
  RF_raw      = p_raw_test,
  RF_Platt    = p_platt_test,
  RF_Isotonic = p_iso_test
)

rows <- lapply(names(models), function(nm){
  p <- models[[nm]]
  auc_point   <- auc_fun(y_test, p)
  brier_point <- brier_fun(y_test, p)
  ci_auc   <- boot_ci(y_test, p, auc_fun)
  ci_brier <- boot_ci(y_test, p, brier_fun)
  tibble(model = nm,
         set   = "test",
         auc   = auc_point,
         auc_ci_low = ci_auc[1], auc_ci_med = ci_auc[2], auc_ci_high = ci_auc[3],
         brier = brier_point,
         brier_ci_low = ci_brier[1], brier_ci_med = ci_brier[2], brier_ci_high = ci_brier[3])
})

out <- dplyr::bind_rows(rows)
readr::write_csv(out, "outputs/metrics/bootstrap_rf_ci.csv")
cat("OK → outputs/metrics/bootstrap_rf_ci.csv\n")

