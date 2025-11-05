# src/r/03e_logit_refit_with_named_segments.R
# Refit logit con segmenti nominati (reference = "Very low risk") + grafici inferenza

set.seed(42)

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(forcats)
  library(broom); library(ggplot2)
})

# =============== Utility ===============
ensure_dirs <- function() {
  dir.create("reports", showWarnings = FALSE)
  dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)
  dir.create("outputs", showWarnings = FALSE)
  dir.create("outputs/model_artifacts", recursive = TRUE, showWarnings = FALSE)
}
ensure_dirs()

# Carica split con 'segment'; se manca *_with_segment.csv, costruiscilo al volo
load_split_with_segment <- function(split = c("train","val","test")) {
  split <- match.arg(split)
  path_ws <- file.path("data/interim", paste0(split, "_with_segment.csv"))
  if (file.exists(path_ws)) {
    df <- read_csv(path_ws, show_col_types = FALSE)
  } else {
    path_feat <- file.path("data/interim", paste0(split, "_features.csv"))
    path_seg  <- file.path("data/interim", paste0("segments_", split, ".csv"))
    if (!file.exists(path_feat) || !file.exists(path_seg)) {
      stop("Manca uno dei file: ", path_feat, " oppure ", path_seg)
    }
    df <- read_csv(path_feat, show_col_types = FALSE) %>%
      left_join(read_csv(path_seg, show_col_types = FALSE), by = "row_id")
    if (any(is.na(df$segment))) stop("Segment NA found in ", split)
    write_csv(df, path_ws)
  }
  df
}

# =============== 1) Carica dati ===============
train <- load_split_with_segment("train")
val   <- load_split_with_segment("val")
test  <- load_split_with_segment("test")

# Assicura factor
train$segment <- factor(train$segment)
val$segment   <- factor(val$segment,   levels = levels(train$segment))
test$segment  <- factor(test$segment,  levels = levels(train$segment))

# =============== 2) Mappa numeri → nomi parlanti ordinati per rischio ===============
risk_tr <- train %>%
  group_by(segment) %>%
  summarise(n = dplyr::n(), default_rate = mean(default), .groups = "drop") %>%
  arrange(desc(default_rate))  # dal più rischioso al più sicuro

# Etichette (adattabili): 5 livelli dal più rischioso al più sicuro
labels_by_risk <- c("Very high risk", "High risk", "Moderate risk", "Low risk", "Very low risk")
stopifnot(nrow(risk_tr) == length(labels_by_risk))

map_tbl <- risk_tr %>% mutate(label = labels_by_risk)
readr::write_csv(map_tbl, "outputs/segment_label_mapping.csv")

# Applica nomi su train/val/test in nuova colonna 'segment_name'
apply_labels <- function(df, map_tbl) {
  lev_order <- map_tbl$segment
  lab_order <- map_tbl$label
  df$segment      <- factor(df$segment, levels = lev_order)
  df$segment_name <- factor(as.character(df$segment),
                            levels = as.character(lev_order),
                            labels = lab_order)
  df
}
train <- apply_labels(train, map_tbl)
val   <- apply_labels(val,   map_tbl)
test  <- apply_labels(test,  map_tbl)

# Salva versioni “named” (comodo per report)
write_csv(train, "data/interim/train_with_segment_named.csv")
write_csv(val,   "data/interim/val_with_segment_named.csv")
write_csv(test,  "data/interim/test_with_segment_named.csv")

# =============== 3) Definisci formula base e refit con reference scelto ===============
# Reference = cluster più sicuro (= ultima riga di risk_tr in ordine decrescente)
ref_label <- tail(map_tbl$label, 1)           # "Very low risk"
train$segment_name <- relevel(train$segment_name, ref = ref_label)

pay  <- c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")
bill <- paste0("BILL_AMT",1:6)
pamt <- paste0("PAY_AMT",1:6)
eng  <- c("max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend")

form_base <- as.formula(paste(
  "default ~", paste(c("SEX","EDUCATION","MARRIAGE","AGE", pay, bill, pamt, eng), collapse = "+")
))

# Fit logistica con segment_name (reference = Very low risk)
fit_named <- glm(update(form_base, . ~ . + segment_name),
                 data = train, family = binomial())

saveRDS(fit_named, "outputs/model_artifacts/logit_baseline_REF_lowrisk.rds")
cat("OK: modello salvato -> outputs/model_artifacts/logit_baseline_REF_lowrisk.rds\n")

# =============== 4) Tabelle inferenza (coef, OR, CI) ===============
tid <- broom::tidy(fit_named, conf.int = TRUE, conf.level = 0.95, exponentiate = FALSE)
tab_inf <- tid %>%
  mutate(
    odds_ratio   = exp(estimate),
    or_conf_low  = exp(conf.low),
    or_conf_high = exp(conf.high)
  ) %>%
  rename(term_name = term, beta = estimate, se = std.error, z = statistic, p = p.value) %>%
  select(term_name, beta, se, z, p, odds_ratio, or_conf_low, or_conf_high, conf.low, conf.high)

write_csv(tab_inf, "reports/logit_inference_named.csv")

# =============== 5) Forest plot OR (con nomi parlanti) ===============
tab_plot <- tab_inf %>%
  filter(term_name != "(Intercept)") %>%
  filter(is.finite(odds_ratio), is.finite(or_conf_low), is.finite(or_conf_high)) %>%
  arrange(odds_ratio) %>%
  mutate(term_name = factor(term_name, levels = unique(term_name)))

plot_or <- ggplot(tab_plot, aes(x = odds_ratio, y = term_name)) +
  geom_vline(xintercept = 1, color = "grey60", linetype = 2) +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = or_conf_low, xmax = or_conf_high), height = 0.15) +
  scale_x_log10() +
  labs(
    title = "Logistic regression — Odds Ratios (95% CI)  [ref = Very low risk]",
    x = "Odds Ratio (log scale, 1 = no effect)", y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("reports/figures/logit_odds_ratio_forest_named.png", plot_or, width = 9, height = 12, dpi = 150)
cat("OK: figura -> reports/figures/logit_odds_ratio_forest_named.png\n")

# =============== 6) AME (Average Marginal Effects) manuale, con nomi ===============
# Predizioni grezze sul train (coerenti col fit)
p_hat <- as.numeric(predict(fit_named, newdata = train, type = "response"))

# Matrice del modello usata dal fit
X <- model.matrix(fit_named)
# rimuovi intercetta
if ("(Intercept)" %in% colnames(X)) {
  X <- X[, setdiff(colnames(X), "(Intercept)"), drop = FALSE]
}
betas <- coef(fit_named)[colnames(X)]

# euristica: dummy se corrispondono a factor (qui intercettiamo segment_name e categoriche principali)
is_dummy <- grepl("^segment_name", colnames(X)) |
  grepl("^SEX$|^MARRIAGE$|^EDUCATION$", colnames(X))

cont_cols <- colnames(X)[!is_dummy]
ame_cont <- sapply(cont_cols, function(nm) {
  mean(betas[nm] * p_hat * (1 - p_hat), na.rm = TRUE)
})

dummy_cols <- colnames(X)[is_dummy]
ame_dummy <- function(colname) {
  X0 <- X; X1 <- X
  X1[, colname] <- X1[, colname] + 1
  eta0 <- drop(cbind(1, X0) %*% coef(fit_named))
  eta1 <- drop(cbind(1, X1) %*% coef(fit_named))
  p0 <- 1/(1 + exp(-eta0)); p1 <- 1/(1 + exp(-eta1))
  mean(p1 - p0, na.rm = TRUE)
}
ame_dum_vals <- sapply(dummy_cols, ame_dummy)

ame_manual <- tibble::tibble(
  term_name = c(names(ame_cont), names(ame_dum_vals)),
  AME = c(as.numeric(ame_cont), as.numeric(ame_dum_vals))
) %>%
  arrange(desc(abs(AME)))

write_csv(ame_manual, "reports/logit_marginal_effects_manual_named.csv")

plot_ame <- ame_manual %>%
  filter(!is.na(AME), term_name != "(Intercept)") %>%
  slice_max(order_by = abs(AME), n = 25) %>%
  arrange(AME) %>%
  mutate(term_name = factor(term_name, levels = term_name)) %>%
  ggplot(aes(x = AME, y = term_name)) +
  geom_vline(xintercept = 0, color = "grey60", linetype = 2) +
  geom_point(size = 2) +
  labs(
    title = "Average Marginal Effects on PD (manual, top 25)  [ref = Very low risk]",
    x = "Δ PD medio per incremento unitario / cambio livello", y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("reports/figures/logit_marginal_effects_manual_named.png", plot_ame, width = 9, height = 10, dpi = 150)
cat("OK: figura -> reports/figures/logit_marginal_effects_manual_named.png\n")

cat("\n=== DONE: refit con segmenti nominati + grafici salvati ===\n")



