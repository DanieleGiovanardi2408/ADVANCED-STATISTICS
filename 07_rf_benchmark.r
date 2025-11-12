# scripts/07_rf_benchmark.R — Random Forest benchmark + file _with_probs_rf
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ranger); library(pROC); library(ggplot2)
})

set.seed(42)
dir.create("outputs/metrics", TRUE, TRUE)
dir.create("outputs/figures", TRUE, TRUE)

# -------- helpers --------
pick_file <- function(...) { x <- c(...); x[file.exists(x)][1] }

find_target <- function(nms) {
  cands <- c("default","y","target","label")
  out <- intersect(cands, nms)[1]
  if (is.na(out)) stop("Target non trovato (default/y/target/label).")
  out
}

is_prob_column <- function(x) {
  if (!is.numeric(x)) return(FALSE)
  q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
  all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
}

find_logit_prob_col <- function(df, ycol) {
  pref <- c("prob_logit","prob","prob_reg","probability","pred_prob",
            "p","yhat","score",".fitted","pred")
  cand <- pref[pref %in% names(df)]
  if (length(cand)) return(cand[1])
  
  numc <- setdiff(names(df)[sapply(df, is.numeric)], ycol)
  if (!length(numc)) return(NA_character_)
  in01 <- vapply(numc, function(nm) is_prob_column(df[[nm]]), logical(1))
  cand2 <- numc[in01]
  if (!length(cand2)) return(NA_character_)
  # prendi quella con AUC più alta
  aucs <- sapply(cand2, function(nm)
    suppressWarnings(as.numeric(auc(roc(df[[ycol]], df[[nm]], quiet=TRUE)))))
  cand2[which.max(aucs)]
}

brier <- function(p,y) mean((p - y)^2)

# -------- carica dati ----------
train_path <- pick_file("data/interim/train_features.csv", "data/interim/train.csv")
val_path   <- pick_file("data/interim/val_features.csv",   "data/interim/val.csv")
test_path  <- pick_file("data/interim/test_features.csv",  "data/interim/test.csv")
if (any(is.na(c(train_path,val_path,test_path)))) {
  stop("Mancano train/val/test in data/interim/ (features.csv o csv base).")
}

train <- suppressMessages(read_csv(train_path, show_col_types = FALSE))
val   <- suppressMessages(read_csv(val_path,   show_col_types = FALSE))
test  <- suppressMessages(read_csv(test_path,  show_col_types = FALSE))

ycol <- find_target(names(train))
if (!all(train[[ycol]] %in% c(0,1)) ||
    !all(val[[ycol]]   %in% c(0,1)) ||
    !all(test[[ycol]]  %in% c(0,1))) stop("Il target deve essere binario (0/1).")

# -------- selezione feature coerenti su tutti i set ----------
common <- Reduce(intersect, list(names(train), names(val), names(test)))
drop_cols <- c(ycol, "id","ID","customer_id","client_id","segment","segment_named")
cand_feats <- setdiff(common, drop_cols)

# tieni solo colonne numeriche (ranger prob con formula gestisce fattori, ma qui standardizziamo)
num_feats <- cand_feats[sapply(train[cand_feats], is.numeric)]
if (!length(num_feats)) {
  warning("Nessuna feature numerica comune trovata; userò tutte le non-target presenti.")
  num_feats <- cand_feats
}
feature_names <- num_feats

# -------- addestra RF (classificazione probabilistica) ----------
train_rf <- train[, c(ycol, feature_names), drop=FALSE]
val_rf   <- val[,   c(ycol, feature_names), drop=FALSE]
test_rf  <- test[,  c(ycol, feature_names), drop=FALSE]

# ranger vuole fattori per classificazione; prob=TRUE restituisce probabilità della classe "1"
train_rf[[ycol]] <- factor(train_rf[[ycol]], levels = c(0,1))

rf <- ranger::ranger(
  formula = reformulate(feature_names, ycol),
  data = train_rf,
  probability = TRUE,
  num.trees = 500,
  mtry = max(1, floor(sqrt(length(feature_names)))),
  min.node.size = 20,
  seed = 42
)

# -------- predizioni ----------
pred_val  <- predict(rf, data = val_rf)$predictions
pred_test <- predict(rf, data = test_rf)$predictions

# estrai colonna della classe "1" robustamente
pick_one_col <- function(mat) {
  if (is.null(dim(mat))) return(as.numeric(mat))
  cn <- colnames(mat)
  if (!is.null(cn) && "1" %in% cn) return(as.numeric(mat[, "1"]))
  # altrimenti prendi la colonna con media più alta (di solito '1')
  as.numeric(mat[, which.max(colMeans(mat)), drop=TRUE])
}
pr_val  <- pick_one_col(pred_val)
pr_test <- pick_one_col(pred_test)

# clamp
pr_val  <- pmin(pmax(pr_val,  1e-6), 1-1e-6)
pr_test <- pmin(pmax(pr_test, 1e-6), 1-1e-6)

# -------- salva file _with_probs_rf per DeLong ----------
write_csv(tibble(!!ycol := val[[ycol]],  prob_rf = pr_val),
          "data/interim/val_with_probs_rf.csv")
write_csv(tibble(!!ycol := test[[ycol]], prob_rf = pr_test),
          "data/interim/test_with_probs_rf.csv")

# -------- metriche ----------
auc_v <- as.numeric(auc(roc(val[[ycol]],  pr_val,  quiet=TRUE)))
auc_t <- as.numeric(auc(roc(test[[ycol]], pr_test, quiet=TRUE)))
met <- tibble(
  set   = c("val","test"),
  auc   = c(auc_v, auc_t),
  brier = c(brier(pr_val, val[[ycol]]), brier(pr_test, test[[ycol]])),
  n     = c(nrow(val), nrow(test))
)
write_csv(met, "outputs/metrics/rf_val_test.csv")
cat("RF AUC val/test:", round(auc_v,3), "/", round(auc_t,3), "\n")

# -------- ROC confronto con Logit se disponibile ----------
logit_path <- pick_file("data/interim/test_with_probs.csv", "data/interim/val_with_probs.csv")
if (!is.na(logit_path)) {
  dfl <- suppressMessages(read_csv(logit_path, show_col_types = FALSE))
  ycol_l <- find_target(names(dfl))
  pl_col <- find_logit_prob_col(dfl, ycol_l)
  if (!is.na(pl_col)) {
    # usa lo stesso split (test o val) per confronto
    if (grepl("test_", basename(logit_path))) {
      y_l  <- dfl[[ycol_l]]; p_l <- dfl[[pl_col]]
      r_l <- roc(y_l, pmin(pmax(p_l, 1e-6),1-1e-6), quiet=TRUE)
      r_r <- roc(test[[ycol]], pr_test, quiet=TRUE)
      title <- "ROC – Test set (Logit vs RF)"
    } else {
      y_l  <- dfl[[ycol_l]]; p_l <- dfl[[pl_col]]
      r_l <- roc(y_l, pmin(pmax(p_l, 1e-6),1-1e-6), quiet=TRUE)
      r_r <- roc(val[[ycol]], pr_val, quiet=TRUE)
      title <- "ROC – Validation set (Logit vs RF)"
    }
    g <- ggplot() +
      geom_line(aes(x = 1 - r_l$specificities, y = r_l$sensitivities), linewidth=1) +
      geom_line(aes(x = 1 - r_r$specificities, y = r_r$sensitivities), linewidth=1, linetype=2) +
      labs(title = title, x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
      theme_minimal(base_size=12)
    out <- "outputs/figures/roc_rf_vs_logit.png"
    ggsave(out, g, width=6.5, height=4.5, dpi=200)
    cat("ROC confronto salvata →", out, "\n")
  } else {
    cat("Probabilità logit non trovate per la ROC di confronto: salto figura.\n")
  }
} else {
  cat("File with_probs (logit) non trovato: salto ROC di confronto.\n")
}
