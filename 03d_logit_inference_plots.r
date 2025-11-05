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

# === 3) VIF (solo per le colonne effettivamente nel modello) ===
# Nota: car::vif lavora su formula e dati; ricostruisco il frame model.matrix
mm <- model.matrix(fit)  # disegno X usato dal modello
df_mm <- as.data.frame(mm)
# rimuovi l'intercetta
if ("(Intercept)" %in% names(df_mm)) df_mm <- df_mm[, setdiff(names(df_mm), "(Intercept)"), drop = FALSE]

# costruiamo una formula artificiale per VIF
form_vif <- as.formula(paste("~", paste(names(df_mm), collapse = " + ")))
v <- tryCatch(car::vif(lm(form_vif, data = df_mm)), error = function(e) NULL)

if (!is.null(v)) {
  vif_tbl <- tibble(term_name = names(v), VIF = as.numeric(v)) %>%
    arrange(desc(VIF))
  readr::write_csv(vif_tbl, "reports/logit_vif.csv")
} else {
  message("VIF non calcolabile (possibile collinearità perfetta).")
}

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
plot_or <- tab_inf %>%
  filter(term_name != "(Intercept)") %>%
  mutate(term_name = fct_reorder(term_name, odds_ratio)) %>%  # ordina per OR
  ggplot(aes(x = odds_ratio, y = term_name)) +
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
plot_ame <- ame_df %>%
  filter(!is.na(AME), term_name != "(Intercept)") %>%
  mutate(term_name = fct_reorder(term_name, abs(AME))) %>%
  ggplot(aes(x = AME, y = term_name)) +
  geom_vline(xintercept = 0, color = "grey60", linetype = 2) +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.15) +
  labs(
    title = "Average Marginal Effects on PD (95% CI)",
    x = "Δ PD (on average) per unit change / level change", y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("reports/figures/logit_marginal_effects.png", plot_ame, width = 9, height = 12, dpi = 150)

cat("OK: wrote reports/logit_inference.csv, reports/logit_vif.csv, reports/logit_marginal_effects.csv\n")
cat("OK: figures in reports/figures/: logit_odds_ratio_forest.png, logit_marginal_effects.png\n")
