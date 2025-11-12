# src/r/03d_logit_inference_plots.R
set.seed(42)

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(forcats)
  library(broom); library(ggplot2)
  library(car)         # VIF
  library(margins)     # AME
})

# === 0) Paths e output dirs ===
dir.create("reports", showWarnings = FALSE)
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

# === 1) Carica modello e dati train (con segment) ===
fit_path  <- "outputs/model_artifacts/logit_baseline.rds"
train_ws  <- "data/interim/train_with_segment.csv"

if (!file.exists(fit_path)) stop("Modello non trovato: ", fit_path)
if (!file.exists(train_ws)) stop("train_with_segment mancante: ", train_ws)

fit   <- readRDS(fit_path)
train <- read_csv(train_ws, show_col_types = FALSE)

# Assicura che le factor abbiano i livelli coerenti con il fit
if ("segment" %in% names(train)) train$segment <- factor(train$segment)

# === 2) Tabella inferenziale (coef, OR, CI95%, p) ===
tid <- broom::tidy(fit, conf.int = TRUE, conf.level = 0.95, exponentiate = FALSE)
# Aggiungi OR ed i CI degli OR
tab_inf <- tid %>%
  mutate(
    odds_ratio   = exp(estimate),
    or_conf_low  = exp(conf.low),
    or_conf_high = exp(conf.high)
  ) %>%
  rename(term_name = term, beta = estimate, se = std.error, z = statistic, p = p.value) %>%
  select(term_name, beta, se, z, p, odds_ratio, or_conf_low, or_conf_high, conf.low, conf.high)

readr::write_csv(tab_inf, "reports/logit_inference.csv")

# === 3) VIF robusto senza formule (niente car::vif) ===
# Prendo il modello design matrix usato dal fit
X <- model.matrix(fit)
# rimuovo l'intercetta
if ("(Intercept)" %in% colnames(X)) {
  X <- X[, setdiff(colnames(X), "(Intercept)"), drop = FALSE]
}

# Calcolo VIF per ogni colonna: regressione X_j ~ X_-j
vif_vec <- sapply(seq_len(ncol(X)), function(j) {
  yj <- X[, j]
  Xj <- X[, -j, drop = FALSE]
  # lm con formula minima per evitare problemi di nomi
  r2 <- summary(lm(yj ~ Xj))$r.squared
  1 / (1 - r2)
})

vif_tbl <- tibble::tibble(
  term_name = colnames(X),
  VIF = as.numeric(vif_vec)
) |> dplyr::arrange(dplyr::desc(VIF))

readr::write_csv(vif_tbl, "reports/logit_vif.csv")
cat("OK: reports/logit_vif.csv scritto\n")


# === 4) Average Marginal Effects (AME) ===
# AME = media, sul campione, degli effetti marginali dP(Y=1)/dX
# Per categoriche, AME è la variazione media nel passaggio di livello (rispetto alla reference)
marg <- margins::margins(fit, data = train)
ame_df <- summary(marg) %>%
  as_tibble() %>%
  rename(term_name = factor, AME = AME, se = SE, z = z, p = p, conf_low = lower, conf_high = upper) %>%
  arrange(desc(abs(AME)))

readr::write_csv(ame_df, "reports/logit_marginal_effects.csv")

# === 5) Grafico 1: Forest degli Odds Ratio (escludi l'intercetta) ===
# --- Forest plot Odds Ratio (robusto) ---
tab_plot <- tab_inf %>%
  filter(term_name != "(Intercept)") %>%
  # rimuovi termini problematici (NA/Inf)
  filter(is.finite(odds_ratio),
         is.finite(or_conf_low),
         is.finite(or_conf_high)) %>%
  arrange(odds_ratio) %>%
  # ordina "a mano": fattore con livelli = ordine corrente
  mutate(term_name = factor(term_name, levels = unique(term_name)))

plot_or <- ggplot(tab_plot, aes(x = odds_ratio, y = term_name)) +
  geom_vline(xintercept = 1, color = "grey60", linetype = 2) +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = or_conf_low, xmax = or_conf_high), height = 0.15) +
  scale_x_log10() +
  labs(
    title = "Logistic regression — Odds Ratios (95% CI)",
    x = "Odds Ratio (log scale, 1 = no effect)", y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("reports/figures/logit_odds_ratio_forest.png", plot_or, width = 9, height = 12, dpi = 150)


# === 6) Grafico 2: AME (con CI) — variabili ordinate per |AME| ===
# Mantieni solo termini informativi (escludi intercept, termini non stimati)
# --- AME senza 'margins' ---
# Predizioni grezze sul train (coerenti col fit)
p_hat <- as.numeric(predict(fit, newdata = train, type = "response"))

# Matrice X del modello (stessa del fit) senza intercetta
X <- model.matrix(fit)
if ("(Intercept)" %in% colnames(X)) {
  X <- X[, setdiff(colnames(X), "(Intercept)"), drop = FALSE]
}
betas <- coef(fit)[colnames(X)]  # allineo i beta alle colonne

# AME per continue: mean( beta_j * p*(1-p) )
# Heuristica: consideriamo "continue" le colonne non-dummy (no pattern 'segment', no 'SEX', etc.)
is_dummy <- grepl("^segment", colnames(X)) | grepl("^SEX$|^MARRIAGE$|^EDUCATION$", colnames(X))
cont_cols <- colnames(X)[!is_dummy]

ame_cont <- sapply(cont_cols, function(nm) {
  mean(betas[nm] * p_hat * (1 - p_hat), na.rm = TRUE)
})

# AME per dummy: discrete change (1 -> 0)
dummy_cols <- colnames(X)[is_dummy]
# funzione per AME discreto su una dummy (tenendo fermi gli altri regressori)
ame_dummy <- function(colname) {
  # stato attuale
  X0 <- X
  X1 <- X
  # per dummy codificata come colonna singola del model.matrix
  X1[, colname] <- X1[, colname] + 1
  # attenzione: con più dummy per una stessa factor può essere più complesso
  # (qui assumiamo codifica 0/1 per quella colonna specifica)
  eta0 <- drop(cbind(1, X0) %*% coef(fit))  # aggiungo intercetta
  eta1 <- drop(cbind(1, X1) %*% coef(fit))
  p0 <- 1/(1+exp(-eta0)); p1 <- 1/(1+exp(-eta1))
  mean(p1 - p0, na.rm = TRUE)
}

# Per semplicità: calcola per le dummy principali che ti interessano
ame_dum_vals <- sapply(dummy_cols, ame_dummy)

ame_manual <- tibble::tibble(
  term_name = c(names(ame_cont), names(ame_dum_vals)),
  AME = c(as.numeric(ame_cont), as.numeric(ame_dum_vals))
) %>% dplyr::arrange(dplyr::desc(abs(AME)))

readr::write_csv(ame_manual, "reports/logit_marginal_effects_manual.csv")

# --- Plot AME robusto ---
plot_ame <- ame_manual %>%
  filter(!is.na(AME), term_name != "(Intercept)") %>%
  dplyr::slice_max(order_by = abs(AME), n = 25) %>%          # top-25 più leggibili
  dplyr::arrange(AME) %>%
  dplyr::mutate(term_name = factor(term_name, levels = term_name)) %>%
  ggplot(aes(x = AME, y = term_name)) +
  geom_vline(xintercept = 0, color = "grey60", linetype = 2) +
  geom_point(size = 2) +
  labs(
    title = "Average Marginal Effects on PD (manual, top 25)",
    x = "Δ PD medio per incremento unitario / cambio livello", y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("reports/figures/logit_marginal_effects_manual.png", plot_ame, width = 9, height = 10, dpi = 150)
