# scripts/04b_auc_bootstrap_delong.R — robusto con diagnostica
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(pROC)
})

set.seed(1)
dir.create("outputs/metrics", TRUE, TRUE)

# --- carica test/val ---
cands <- c("data/interim/test_with_probs.csv", "data/interim/val_with_probs.csv")
path  <- cands[file.exists(cands)][1]
if (is.na(path)) stop("Manca test_with_probs.csv/val_with_probs.csv in data/interim/")
df <- suppressMessages(read_csv(path, show_col_types = FALSE))
cat("File usato:", path, "\n")

# --- target ---
target_candidates <- c("default","y","target","label")
y_col <- intersect(target_candidates, names(df))[1]
if (is.na(y_col)) stop("Target non trovato (default/y/target/label).")
y_raw <- df[[y_col]]
if (!all(y_raw %in% c(0,1))) stop("Il target deve essere 0/1.")
y_raw <- as.integer(y_raw)

# helper: AUC sicura su una colonna x
safe_auc <- function(y, x){
  # keep finite & non-missing
  ok <- is.finite(x) & !is.na(x) & is.finite(y) & !is.na(y)
  y <- y[ok]; x <- x[ok]
  # range clamp (probabilità)
  x <- pmin(pmax(x, 0), 1)
  if (length(x) < 30L)     return(NA_real_)  # troppo pochi
  if (sd(x) == 0)          return(NA_real_)  # costante
  # entrambe le classi presenti
  if (length(unique(y)) < 2) return(NA_real_)
  out <- tryCatch({
    as.numeric(auc(roc(response = y, predictor = x, quiet = TRUE)))
  }, error = function(e) NA_real_)
  out
}

# --- trova la colonna con le probabilità del logit (robusto) ---
find_prob_col <- function(df, y_col){
  preferred <- c("prob_logit","prob","prob_reg","probability","pred_prob",
                 "p","yhat","score",".fitted","pred")
  cand <- preferred[preferred %in% names(df)]
  if (length(cand)) return(cand[1])
  
  numc <- names(df)[sapply(df, is.numeric)]
  numc <- setdiff(numc, y_col)
  if (!length(numc)) return(NA_character_)
  
  # filtra a [0,1] (entro margine), non costanti, abbastanza dati
  in01 <- vapply(numc, function(nm){
    x <- df[[nm]]
    q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
  }, logical(1))
  cand2 <- numc[in01]
  if (!length(cand2)) return(NA_character_)
  
  # scegli quella con AUC maggiore (safe)
  aucs <- sapply(cand2, function(nm) safe_auc(df[[y_col]], df[[nm]]))
  if (all(is.na(aucs))) return(NA_character_)
  cand2[which.max(aucs)]
}

pl_col <- find_prob_col(df, y_col)

# Diagnostica se non trovata
if (is.na(pl_col)) {
  numc <- setdiff(names(df)[sapply(df, is.numeric)], y_col)
  info <- lapply(numc, function(nm){
    x <- df[[nm]]
    q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    data.frame(
      col = nm, n_non_na = sum(is.finite(x)), sd = suppressWarnings(sd(x, na.rm=TRUE)),
      q01 = q[1], q99 = q[2], auc_try = safe_auc(df[[y_col]], x)
    )
  })
  diag <- dplyr::bind_rows(info)
  out_diag <- "outputs/metrics/_prob_column_diagnostic.csv"
  write_csv(diag, out_diag)
  stop("Colonna probabilità non trovata automaticamente.\n",
       "Controlla ", out_diag, " e rinomina la tua colonna a 'prob' o 'prob_logit'.")
}

cat("Colonna prob scelta:", pl_col, "\n")
p_raw <- df[[pl_col]]

# prepara vettori puliti per ROC
ok <- is.finite(p_raw) & !is.na(p_raw) & is.finite(y_raw) & !is.na(y_raw)
y <- y_raw[ok]
p <- pmin(pmax(p_raw[ok], 1e-6), 1-1e-6)

# Se sembrano "prob di NON default", inverti
roc_tmp <- tryCatch(roc(y, p, quiet = TRUE), error = function(e) NULL)
auc_tmp <- if (is.null(roc_tmp)) NA_real_ else as.numeric(auc(roc_tmp))
flipped <- FALSE
if (!is.na(auc_tmp) && auc_tmp < 0.5) { p <- 1 - p; flipped <- TRUE }

# --- AUC + CI bootstrap ---
roc_l <- roc(response = y, predictor = p, quiet = TRUE)
ci_l  <- ci.auc(roc_l, conf.level = 0.95, method = "bootstrap", boot.n = 2000)

auc_tbl <- tibble(
  set = ifelse(grepl("test_", basename(path)), "test","val"),
  model = "logit",
  prob_column = pl_col,
  flipped = flipped,
  n = length(y),
  auc = as.numeric(auc(roc_l)),
  ci_low = ci_l[1], ci_med = ci_l[2], ci_high = ci_l[3],
  method = "bootstrap"
)
write_csv(auc_tbl, "outputs/metrics/auc_ci_test.csv")
cat("AUC logit:", round(auc_tbl$auc,3),
    " CI95%[", round(auc_tbl$ci_low,3), ",", round(auc_tbl$ci_high,3), "]  n=", auc_tbl$n, "\n")

# --- DeLong vs RF se disponibile ---
rf_prob_col <- c("prob_rf","rf_prob","rf_score")
rf_col <- rf_prob_col[rf_prob_col %in% names(df)][1]

if (is.na(rf_col)) {
  rf_path <- gsub("_with_probs\\.csv$", "_with_probs_rf.csv", path)
  if (file.exists(rf_path)) {
    dfr <- suppressMessages(read_csv(rf_path, show_col_types = FALSE))
    rf_col <- rf_prob_col[rf_prob_col %in% names(dfr)][1]
    if (!is.na(rf_col)) df[[rf_col]] <- dfr[[rf_col]]
  }
}

if (!is.na(rf_col)) {
  pr <- df[[rf_col]]
  ok2 <- is.finite(pr) & !is.na(pr) & is.finite(y_raw) & !is.na(y_raw)
  y2  <- y_raw[ok2]; pr2 <- pmin(pmax(pr[ok2], 1e-6), 1-1e-6)
  r_l <- roc(y2, p[match(which(ok2), which(ok))], quiet=TRUE)
  r_r <- roc(y2, pr2, quiet=TRUE)
  del <- roc.test(r_l, r_r, method = "delong")
  outd <- tibble(
    set  = ifelse(grepl("test_", basename(path)), "test","val"),
    test = "DeLong",
    auc_logit = as.numeric(auc(r_l)),
    auc_rf    = as.numeric(auc(r_r)),
    p_value   = as.numeric(del$p.value)
  )
  write_csv(outd, "outputs/metrics/delong_logit_vs_rf.csv")
  cat("DeLong logit vs RF → p =", signif(outd$p_value,3), "\n")
} else {
  cat("Nessuna curva RF trovata: salto DeLong.\n")
}


