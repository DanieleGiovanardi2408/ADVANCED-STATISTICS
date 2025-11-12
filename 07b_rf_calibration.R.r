# scripts/07b_rf_calibration.R — Platt vs Isotonic sul RF + confronto Brier
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(isotone); library(pROC); library(ggplot2)
})
dir.create("outputs/metrics", TRUE, TRUE); dir.create("outputs/figures", TRUE, TRUE)

pick <- function(...) { x <- c(...); x[file.exists(x)][1] }
find_target <- function(nms) intersect(c("default","y","target","label"), nms)[1]
brier <- function(p,y) mean((p-y)^2)

# carica RF preds (già creati da 07_rf_benchmark.R)
va_rf <- pick("data/interim/val_with_probs_rf.csv"); te_rf <- pick("data/interim/test_with_probs_rf.csv")
stopifnot(!is.na(va_rf), !is.na(te_rf))
V <- suppressMessages(read_csv(va_rf, show_col_types = FALSE))
T <- suppressMessages(read_csv(te_rf, show_col_types = FALSE))
ycol <- find_target(names(V)); stopifnot(!is.na(ycol))

y_v <- as.integer(V[[ycol]]); p_v <- pmin(pmax(as.numeric(V$prob_rf), 1e-8), 1-1e-8)
y_t <- as.integer(T[[ycol]]); p_t <- pmin(pmax(as.numeric(T$prob_rf), 1e-8), 1-1e-8)

# --- Platt (logit su val) ---
platt <- glm(y_v ~ qlogis(p_v), family=binomial())
p_t_platt <- pmin(pmax(as.numeric(plogis(predict(platt, newdata=data.frame(p_v=p_t)))), 1e-8), 1-1e-8)

# --- Isotonic (val) ---
iso <- isoreg(p_v, y_v)  # calibra su val
f_iso <- stats::approxfun(iso$x, pmin(pmax(iso$yf, 0), 1), rule=2)
p_t_iso <- pmin(pmax(f_iso(p_t), 1e-8), 1-1e-8)

# metriche su test
auc_raw  <- as.numeric(auc(roc(y_t, p_t,       quiet=TRUE)))
auc_pl   <- as.numeric(auc(roc(y_t, p_t_platt, quiet=TRUE)))
auc_iso  <- as.numeric(auc(roc(y_t, p_t_iso,   quiet=TRUE)))
br_raw   <- brier(p_t,       y_t)
br_pl    <- brier(p_t_platt, y_t)
br_iso   <- brier(p_t_iso,   y_t)

out <- tibble(
  model = c("RF raw","RF Platt","RF Isotonic"),
  auc_test = c(auc_raw, auc_pl, auc_iso),
  brier_test = c(br_raw, br_pl, br_iso)
)
readr::write_csv(out, "outputs/metrics/rf_calibration_compare.csv")
print(out)

# figura calibrazione
mkcal <- function(p,y,lab){ cutp <- cut(p, breaks=quantile(p, probs=seq(0,1,length.out=11)), include.lowest=TRUE)
df <- data.frame(y=y, p=p, b=cutp) %>% group_by(b) %>% summarise(n=n(), p_bar=mean(p), o_bar=mean(y), .groups="drop")
df$lab <- lab; df }
C <- dplyr::bind_rows(mkcal(p_t,y_t,"RF raw"), mkcal(p_t_platt,y_t,"RF Platt"), mkcal(p_t_iso,y_t,"RF Isotonic"))
g <- ggplot(C, aes(x=p_bar, y=o_bar, shape=lab, size=n)) +
  geom_point(alpha=.85) + geom_abline(slope=1, intercept=0, linetype=2) +
  labs(title="Calibration — RF (raw vs post-hoc)", x="Predicted PD (bin mean)", y="Observed rate") +
  scale_size_area(max_size=8) + theme_minimal(base_size=12)
ggsave("outputs/figures/calibration_rf_pre_post.png", g, width=6.5, height=4.8, dpi=200)
