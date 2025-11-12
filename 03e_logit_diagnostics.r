# scripts/03e_logit_diagnostics.R — Hosmer–Lemeshow (robusto) + influenza (light)
suppressPackageStartupMessages({
  library(readr); library(dplyr)
  library(glmnet)   # per predict.glmnet / predict.cv.glmnet
})

dir.create("outputs/metrics", TRUE, TRUE)

cat("\n=== 03e: Logit diagnostics (HL + influence) ===\n")

# -------- 1) Carica modello logit --------
model_candidates <- c(
  "outputs/model_artifacts/logit_reg_best.rds",             # cv.glmnet (lambda.min)
  "outputs/model_artifacts/logit_reg_best_auc.rds",         # glmnet (lognet)
  "outputs/model_artifacts/logit_baseline_REF_lowrisk.rds", # glm
  "outputs/model_artifacts/logit_baseline.rds"              # glm
)
model_path <- model_candidates[file.exists(model_candidates)][1]
if (is.na(model_path)) stop("Modello logit non trovato in outputs/model_artifacts/")
m <- readRDS(model_path)
is_glm      <- inherits(m, "glm")
is_cvglmnet <- inherits(m, "cv.glmnet")
is_glmnet   <- inherits(m, "glmnet")
cat("Model:", basename(model_path), "| class:", paste(class(m), collapse=", "), "\n")

# -------- helper: allineamento feature e pred_fun --------
align_matrix <- function(df, want_cols){
  X <- as.matrix(df[, intersect(want_cols, names(df)), drop=FALSE])
  miss <- setdiff(want_cols, colnames(X))
  if (length(miss)) {
    X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(miss),
                         dimnames=list(NULL, miss)))
  }
  X[, want_cols, drop=FALSE]
}

pred_fun <- if (is_cvglmnet) {
  function(object, newdata){
    want <- rownames(object$glmnet.fit$beta)
    X <- align_matrix(newdata, want)
    as.numeric(glmnet::predict.glmnet(object$glmnet.fit, newx = X,
                                      type = "response", s = object$lambda.min))
  }
} else if (is_glmnet) {  # classi "lognet","glmnet"
  function(object, newdata){
    want <- rownames(object$beta)
    X <- align_matrix(newdata, want)
    # usa il lambda col dev.ratio massimo
    s_use <- object$lambda[ which.max(object$dev.ratio) ]
    as.numeric(glmnet::predict.glmnet(object, newx = X,
                                      type = "response", s = s_use))
  }
} else if (is_glm) {
  function(object, newdata){
    as.numeric(stats::predict(object, newdata = newdata, type = "response"))
  }
} else {
  stop("Classe modello non supportata: ", paste(class(m), collapse=", "))
}

# quali predittori servono?
feat_names <- if (is_glm) {
  resp <- as.character(formula(m)[[2]])
  setdiff(all.vars(formula(m)), resp)
} else if (is_cvglmnet) {
  rownames(m$glmnet.fit$beta)
} else {
  rownames(m$beta)
}

# -------- 2) Dati per HL: preferisci TEST, poi VAL --------
pick_file <- function(...) { x <- c(...); x[file.exists(x)][1] }
path_x <- pick_file("data/interim/test_features.csv", "data/interim/test.csv",
                    "data/interim/val_features.csv",  "data/interim/val.csv")
if (is.na(path_x)) stop("Mancano test/val in data/interim/")
X <- suppressMessages(read_csv(path_x, show_col_types = FALSE))

y_col <- intersect(c("default","y","target","label"), names(X))[1]
if (is.na(y_col)) stop("Target non trovato in ", basename(path_x))
stopifnot(all(X[[y_col]] %in% c(0,1)))

# fattori “segment” se presenti e modello GLM li usa
if (is_glm) {
  if ("segment" %in% feat_names && "segment" %in% names(X)) X$segment <- factor(X$segment)
  if ("segment_named" %in% feat_names && "segment_named" %in% names(X)) X$segment_named <- factor(X$segment_named)
}

# tieni solo colonne utili (feature + target)
keep <- unique(c(feat_names, y_col))
Xuse <- X[, intersect(keep, names(X)), drop=FALSE]

# probabilità (clamp per sicurezza)
p <- pred_fun(m, Xuse)
p <- pmin(pmax(p, 1e-6), 1-1e-6)
y <- as.integer(Xuse[[y_col]])

# -------- 3) Hosmer–Lemeshow (g=10) — versione robusta a ties --------
make_hl_bins <- function(p, y, g = 10L){
  # se poche probabilità distinte, riduci g
  u <- length(unique(p))
  g_use <- max(2L, min(g, u))
  # decili per rango (evita breaks non unici)
  rk <- rank(p, ties.method = "average")
  bin <- ceiling(rk / length(p) * g_use)
  
  d <- dplyr::tibble(p = p, y = y, bin = bin) |>
    dplyr::group_by(bin) |>
    dplyr::summarise(n = dplyr::n(),
                     obs = sum(y),
                     exp = sum(p),
                     p_bar = mean(p),
                     .groups = "drop")
  attr(d, "g_use") <- g_use
  d
}

hl_tbl <- make_hl_bins(p, y, g = 10L)
g_eff  <- attr(hl_tbl, "g_use")

# statistica HL classica: sum ( (O-E)^2 / (E * (1 - E/n)) )
HL_stat <- sum( (hl_tbl$obs - hl_tbl$exp)^2 /
                  (pmax(hl_tbl$exp, 1e-12) * (1 - (hl_tbl$exp / pmax(hl_tbl$n,1)))) )
HL_df   <- g_eff - 2L
HL_p    <- if (HL_df > 0) 1 - pchisq(HL_stat, df = HL_df) else NA_real_

out_hl <- sprintf(
  "Hosmer–Lemeshow (rank-binned) on %s\nN=%d  g=%d  statistic=%.3f  df=%d  p=%s\nNota: uso bin per rango per evitare breaks duplicati con molte tie nelle PD.",
  ifelse(grepl("test", path_x), "TEST", "VAL"),
  length(p), g_eff, HL_stat, HL_df,
  ifelse(is.na(HL_p), "NA", sprintf("%.4g", HL_p))
)

readr::write_csv(hl_tbl, "outputs/metrics/hl_bins.csv")
writeLines(out_hl, "outputs/metrics/hl_test.txt")
cat(out_hl, "\nSaved: outputs/metrics/hl_test.txt, hl_bins.csv\n")

# -------- 4) Influenza / outlier (solo GLM su TRAIN) --------
if (is_glm) {
  train_path <- pick_file("data/interim/train_features.csv", "data/interim/train.csv")
  if (!is.na(train_path)) {
    Tr <- suppressMessages(read_csv(train_path, show_col_types = FALSE))
    stopifnot(any(names(Tr) == y_col))
    if ("segment" %in% feat_names && "segment" %in% names(Tr)) Tr$segment <- factor(Tr$segment)
    if ("segment_named" %in% feat_names && "segment_named" %in% names(Tr)) Tr$segment_named <- factor(Tr$segment_named)
    Tr_use <- Tr[, intersect(unique(c(feat_names, y_col)), names(Tr)), drop=FALSE]
    
    # rifit GLM solo per avere hatvalues & influence coerenti (stessa formula)
    fml <- as.formula(paste(y_col, "~", paste(feat_names, collapse = "+")))
    m_train <- glm(fml, data = Tr_use, family = binomial())
    
    hat  <- hatvalues(m_train)
    cd   <- cooks.distance(m_train)
    rstd <- rstandard(m_train, type = "deviance")
    db   <- as.data.frame(dfbetas(m_train))
    db_max <- apply(abs(db), 1, max, na.rm = TRUE)
    
    id_col <- intersect(c("id","ID","customer_id","client_id"), names(Tr_use))
    id <- if (length(id_col)) Tr_use[[id_col[1]]] else seq_len(nrow(Tr_use))
    
    infl <- tibble(
      id = id,
      hat = hat,
      cooks_d = cd,
      rstandard = rstd,
      dfbeta_max = db_max
    ) |>
      mutate(flag_hat   = hat   > (3*mean(hat, na.rm=TRUE)),
             flag_cook  = cooks_d > (4/length(hat)),
             flag_rstd  = abs(rstandard) > 3,
             flag_dfb   = dfbeta_max > 1) |>
      arrange(desc(cooks_d))
    
    write_csv(infl, "outputs/metrics/influence_summary.csv")
    cat("Saved: outputs/metrics/influence_summary.csv (ordinato per Cook's D)\n")
  } else {
    cat("Train non trovato: salto influenza.\n")
  }
} else {
  cat("Modello non-GLM: influenza classica non applicabile (salto sezione).\n")
}

cat("=== DONE diagnostics ===\n")




suppressPackageStartupMessages({library(readr); library(dplyr); library(pROC); library(glmnet)})

# --- helper robusto per trovare la colonna probabilità ---
find_prob_col <- function(df, ycol) {
  preferred <- c("prob_logit","prob","prob_reg","probability","pred_prob","p","yhat","score",".fitted","pred")
  cand <- preferred[preferred %in% names(df)]
  if (length(cand)) return(cand[1])
  
  numc <- setdiff(names(df)[sapply(df, is.numeric)], ycol)
  if (!length(numc)) return(NA_character_)
  
  in01 <- vapply(numc, function(nm){
    x <- df[[nm]]
    q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
  }, logical(1))
  cand2 <- numc[in01]
  if (!length(cand2)) return(NA_character_)
  
  aucs <- sapply(cand2, function(nm){
    suppressWarnings(as.numeric(auc(roc(df[[ycol]], df[[nm]], quiet=TRUE))))
  })
  cand2[which.max(aucs)]
}

# --- A) PD dal file with_probs ---
wp <- c("data/interim/test_with_probs.csv", "data/interim/val_with_probs.csv")
wp <- wp[file.exists(wp)][1]
stopifnot(!is.na(wp))
dl <- readr::read_csv(wp, show_col_types = FALSE)

ycol <- intersect(c("default","y","target","label"), names(dl))[1]
stopifnot(!is.na(ycol))
pl_col <- find_prob_col(dl, ycol)
if (is.na(pl_col)) {
  # diagnostica salvata se non trova nulla
  numc <- names(dl)[sapply(dl, is.numeric)]
  diag <- dplyr::summarise(dplyr::across(dl[numc], ~{
    q <- suppressWarnings(quantile(.x, c(.01,.99), na.rm=TRUE))
    c(n_non_na = sum(is.finite(.x)), sd = suppressWarnings(sd(.x, na.rm=TRUE)),
      q01 = q[1], q99 = q[2])
  }))
  readr::write_csv(tibble(col=numc, t(diag)), "outputs/metrics/_prob_column_diagnostic_with_probs.csv")
  stop("Non trovo la colonna probabilità nel file ", basename(wp),
       ". Vedi outputs/metrics/_prob_column_diagnostic_with_probs.csv e rinomina la colonna a 'prob'.")
}
y_file <- dl[[ycol]]
p_file <- pmin(pmax(dl[[pl_col]], 1e-6), 1-1e-6)
cat("with_probs:", basename(wp), "| prob_col:", pl_col,
    "| range:[", sprintf("%.4f, %.4f", min(p_file), max(p_file)), "]",
    "| AUC=", round(as.numeric(auc(roc(y_file, p_file, quiet=TRUE))),3), "\n", sep=" ")

# --- B) PD dal modello (stesso split del file) ---
model_candidates <- c(
  "outputs/model_artifacts/logit_reg_best.rds",
  "outputs/model_artifacts/logit_reg_best_auc.rds",
  "outputs/model_artifacts/logit_baseline_REF_lowrisk.rds",
  "outputs/model_artifacts/logit_baseline.rds"
)
model_path <- model_candidates[file.exists(model_candidates)][1]
stopifnot(!is.na(model_path))
m <- readRDS(model_path)

# carico features coerenti con lo split
# carico features coerenti con lo split
if (grepl("test_", basename(wp))) {
  fx_candidates <- c("data/interim/test_features.csv", "data/interim/test.csv")
} else {
  fx_candidates <- c("data/interim/val_features.csv", "data/interim/val.csv")
}
pick_file <- function(...) { x <- c(...); x[file.exists(x)][1] }
fx <- pick_file(fx_candidates)
stopifnot(!is.na(fx))
TX <- readr::read_csv(fx, show_col_types = FALSE)


feat_names <- if (inherits(m,"glm")) {
  resp <- as.character(formula(m)[[2]]); setdiff(all.vars(formula(m)), resp)
} else if (inherits(m,"cv.glmnet")) rownames(m$glmnet.fit$beta) else rownames(m$beta)

align_matrix <- function(df, want){
  X <- as.matrix(df[, intersect(want, names(df)), drop=FALSE])
  miss <- setdiff(want, colnames(X))
  if (length(miss)) X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(miss), dimnames=list(NULL, miss)))
  X[, want, drop=FALSE]
}

pred_fun <- if (inherits(m,"cv.glmnet")) {
  function(m,newdata){ X<-align_matrix(newdata, rownames(m$glmnet.fit$beta)); as.numeric(predict(m, newx=X, type="response", s="lambda.min")) }
} else if (inherits(m,"glmnet")) {
  function(m,newdata){ X<-align_matrix(newdata, rownames(m$beta)); s_use<-m$lambda[which.max(m$dev.ratio)]; as.numeric(predict(m, newx=X, type="response", s=s_use)) }
} else {
  function(m,newdata){ as.numeric(predict(m, newdata=newdata, type="response")) }
}

p_model <- pmin(pmax(pred_fun(m, TX), 1e-6), 1-1e-6)
cat("model.predict | range:[", sprintf("%.4f, %.4f", min(p_model), max(p_model)), "]",
    "| AUC=", round(as.numeric(auc(roc(TX[[ycol]], p_model, quiet=TRUE))),3), "\n", sep=" ")

cat("cor(P_file, P_model) =", sprintf("%.3f", suppressWarnings(cor(p_file, p_model, use="complete.obs"))), "\n")


