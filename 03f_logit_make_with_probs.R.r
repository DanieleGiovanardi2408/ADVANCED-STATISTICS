# scripts/03f_logit_make_with_probs.R
suppressPackageStartupMessages({
  library(readr); library(dplyr)
})

cat("\n=== MAKE with_probs (logit) ===\n")
set.seed(1)

# -- helper: scegli primo file esistente
pick_first <- function(...) { x <- c(...); x[file.exists(x)][1] }

# -- carica modello logit (preferisci glmnet "best")
model_candidates <- c(
  "outputs/model_artifacts/logit_reg_best.rds",
  "outputs/model_artifacts/logit_reg_best_auc.rds",
  "outputs/model_artifacts/logit_baseline.rds",
  "outputs/model_artifacts/logit_baseline_REF_lowrisk.rds"
)
model_path <- pick_first(model_candidates)
if (is.na(model_path)) stop("Modello logit non trovato in outputs/model_artifacts/")
m <- readRDS(model_path)
is_glm      <- inherits(m, "glm")
is_cvglmnet <- inherits(m, "cv.glmnet")
is_glmnet   <- inherits(m, "glmnet") || inherits(m, "lognet")
cat("Model ->", basename(model_path), "| class:", paste(class(m), collapse=", "), "\n")

# -- feature richieste dal modello
if (is_glm) {
  resp       <- as.character(formula(m)[[2]])
  feat_names <- setdiff(all.vars(formula(m)), resp)
} else if (is_cvglmnet) {
  feat_names <- rownames(m$glmnet.fit$beta)
} else if (is_glmnet) {
  feat_names <- rownames(m$beta)
} else stop("Classe modello non supportata:", paste(class(m), collapse=", "))

# -- funzione per allineare le colonne a glmnet
align_matrix <- function(df, want_cols){
  X <- as.matrix(df[, intersect(want_cols, names(df)), drop=FALSE])
  miss <- setdiff(want_cols, colnames(X))
  if (length(miss)) {
    X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(miss),
                         dimnames = list(NULL, miss)))
  }
  X[, want_cols, drop=FALSE]
}

# -- pred.fun che restituisce prob di default
pred_fun <- if (is_cvglmnet) {
  function(object, newdata){
    X <- align_matrix(newdata, rownames(object$glmnet.fit$beta))
    as.numeric(predict(object, newx = X, type = "response", s = "lambda.min"))
  }
} else if (is_glmnet) {
  function(object, newdata){
    X <- align_matrix(newdata, rownames(object$beta))
    as.numeric(predict(object, newx = X, type = "response"))
  }
} else { # glm
  function(object, newdata){
    as.numeric(predict(object, newdata = newdata, type = "response"))
  }
}

# -- individua target nei dati
find_target <- function(nms){
  cands <- c("default","y","target","label")
  out <- intersect(cands, nms)[1]
  if (is.na(out)) stop("Target non trovato (default/y/target/label) nel dataset.")
  out
}

# -- funzione unica: legge split, calcola prob e scrive *_with_probs.csv
make_probs_for <- function(split = c("train","val","test")){
  split <- match.arg(split)
  path <- pick_first(
    file.path("data/interim", paste0(split, "_features.csv")),
    file.path("data/interim", paste0(split, ".csv"))
  )
  if (is.na(path)) { cat("Skip", split, ": file non trovato.\n"); return(invisible(NULL)) }
  
  df <- suppressMessages(read_csv(path, show_col_types = FALSE))
  ycol <- find_target(names(df))
  
  # alias segmenti se servono al GLM
  if (is_glm && "segment" %in% feat_names && !"segment" %in% names(df) && "segment_named" %in% names(df)) {
    df$segment <- df$segment_named
  }
  if (is_glm && "segment" %in% names(df)) df$segment <- factor(df$segment)
  if (is_glm && "segment_named" %in% names(df)) df$segment_named <- factor(df$segment_named)
  
  # verifica feature minime
  need <- intersect(feat_names, names(df))
  if (!length(need)) stop("Nessuna feature del modello trovata in ", basename(path))
  
  # predizioni
  newdata <- df[, unique(c(need)), drop=FALSE]
  p <- pred_fun(m, newdata)
  p <- pmin(pmax(as.numeric(p), 1e-8), 1-1e-8)
  
  out <- tibble(!!ycol := df[[ycol]], prob = p)
  out_path <- file.path("data/interim", paste0(split, "_with_probs.csv"))
  write_csv(out, out_path)
  cat("OK ->", out_path, " | n=", nrow(out), " | prob range:[",
      sprintf("%.4f", min(p)), ", ", sprintf("%.4f", max(p)), "]\n", sep="")
}

dir.create("data/interim", TRUE, TRUE)
invisible(lapply(c("train","val","test"), make_probs_for))

cat("=== DONE make_with_probs ===\n")

