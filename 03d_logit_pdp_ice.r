# scripts/03d_logit_pdp_ice.R — definitivo (PDP/ICE veloci e stabili)
suppressPackageStartupMessages({
  library(pdp); library(ggplot2); library(readr); library(dplyr)
})

cat("\n=== PDP/ICE — run ===\n")

# (Opzionale) porta il WD alla root del progetto se c'è un .Rproj
rp <- list.files(".", pattern="\\.Rproj$", recursive=TRUE, full.names=TRUE)
if (length(rp)) { setwd(dirname(rp[1])); cat("Project root:", getwd(), "\n") }

# 1) Carica il modello (primo disponibile)
model_candidates <- c(
  "outputs/model_artifacts/logit_baseline.rds",
  "outputs/model_artifacts/logit_baseline_REF_lowrisk.rds",
  "outputs/model_artifacts/logit_reg_best.rds",
  "outputs/model_artifacts/logit_reg_best_auc.rds"
)
model_path <- model_candidates[file.exists(model_candidates)][1]
if (is.na(model_path)) stop("Modello non trovato in outputs/model_artifacts/")
m <- readRDS(model_path)
is_glm      <- inherits(m, "glm")
is_cvglmnet <- inherits(m, "cv.glmnet")
is_glmnet   <- inherits(m, "glmnet")
cat("Model:", basename(model_path), "| class:", paste(class(m), collapse=", "), "\n")

# 2) Variabili richieste dal modello (nomi nudi)
if (is_glm) {
  resp        <- as.character(formula(m)[[2]])
  feat_names  <- setdiff(all.vars(formula(m)), resp)
  needed_terms <- feat_names
} else if (is_cvglmnet) {
  needed_terms <- rownames(m$glmnet.fit$beta); feat_names <- needed_terms
} else if (is_glmnet) {
  needed_terms <- rownames(m$beta);            feat_names <- needed_terms
} else stop("Classe modello non supportata: ", paste(class(m), collapse=", "))

need_seg       <- "segment"       %in% feat_names
need_seg_named <- "segment_named" %in% feat_names
xlevels_m      <- if (is_glm) m$xlevels else list()

# 3) Dataset di train da data/interim/
train_candidates <- if (need_seg) {
  c("data/interim/train_with_segment.csv",
    "data/interim/train_with_segment_named.csv",
    "data/interim/train.csv",
    "data/interim/train_features.csv")
} else if (need_seg_named) {
  c("data/interim/train_with_segment_named.csv",
    "data/interim/train_with_segment.csv",
    "data/interim/train.csv",
    "data/interim/train_features.csv")
} else {
  c("data/interim/train_with_segment.csv",
    "data/interim/train_with_segment_named.csv",
    "data/interim/train.csv",
    "data/interim/train_features.csv")
}
train_path <- train_candidates[file.exists(train_candidates)][1]
if (is.na(train_path)) {
  seen <- try(list.files("data/interim", pattern="\\.csv$", full.names=TRUE), silent=TRUE)
  stop("Nessun file train* trovato in data/interim/.",
       if (!inherits(seen,"try-error")) paste0("\nVisti: ", paste(basename(seen), collapse=", ")))
}
train <- suppressMessages(readr::read_csv(train_path, show_col_types = FALSE))
cat("Train:", train_path, "\n")

# 4) Alias e coercizione fattori
if (need_seg && !"segment" %in% names(train) && "segment_named" %in% names(train)) {
  train$segment <- train$segment_named; cat("Alias: segment <- segment_named\n")
}
if (need_seg_named && !"segment_named" %in% names(train) && "segment" %in% names(train)) {
  train$segment_named <- train$segment; cat("Alias: segment_named <- segment\n")
}
if (is_glm && need_seg       && "segment"       %in% names(train))       train$segment       <- factor(train$segment)
if (is_glm && need_seg_named && "segment_named" %in% names(train)) train$segment_named <- factor(train$segment_named)

# 5) Target e controllo feature
target_candidates <- c("default","y","target","label")
target <- intersect(target_candidates, names(train))[1]
drop_exist <- intersect(c(target, "segment", "segment_named", "id", "customer_id"), names(train))
missing_feats <- setdiff(feat_names, names(train))
if (length(missing_feats)) {
  stop("Nel train mancano colonne richieste dal modello: ",
       paste(missing_feats, collapse=", "),
       "\nSuggerimento: rigenera data/interim/train_with_segment*.csv coerente col modello.")
}

# Allinea livelli fattoriali ai livelli del modello (solo glm)
if (is_glm && length(xlevels_m)) {
  for (nm in names(xlevels_m)) {
    lv <- xlevels_m[[nm]]
    if (!(nm %in% names(train))) {
      train[[nm]] <- factor(lv[1], levels = lv)
      cat("Fattore aggiunto:", nm, "->", lv[1], "\n")
    } else {
      train[[nm]] <- factor(as.character(train[[nm]]),
                            levels = union(levels(factor(train[[nm]])), lv))
    }
  }
}

# 6) pred.fun (probabilità)
align_matrix <- function(df, want_cols){
  X <- as.matrix(df[, intersect(want_cols, names(df)), drop=FALSE])
  missing <- setdiff(want_cols, colnames(X))
  if (length(missing)) {
    X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(missing),
                         dimnames=list(NULL, missing)))
  }
  X[, want_cols, drop=FALSE]
}
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
} else {
  function(object, newdata){
    as.numeric(predict(object, newdata = newdata, type = "response"))
  }
}

# 7) Mini-test predict
mini <- train[seq_len(min(200L, nrow(train))), unique(c(feat_names, target)), drop=FALSE]
test_pred <- try(pred_fun(m, mini), silent=TRUE)
if (inherits(test_pred, "try-error")) stop("predict() fallisce: ", as.character(test_pred))
cat("Predict OK su mini-batch (n=", nrow(mini), ")\n", sep="")

# 8) Variabili PDP
preferred <- c("util_last","pay_ratio","cnt_dpd")
vars <- preferred[preferred %in% feat_names]
if (length(vars) == 0L) {
  numc <- feat_names[feat_names %in% names(train)[sapply(train, is.numeric)]]
  vars <- setdiff(numc, drop_exist)[1:min(3, length(setdiff(numc, drop_exist)))]
}
if (!length(vars)) stop("Nessuna variabile numerica adatta per PDP tra le feature del modello.")

dir.create("outputs/figures", TRUE, TRUE)
dir.create("outputs/metrics", TRUE, TRUE)

# 9) Funzione PDP “light” (niente loess; linea media senza smoothing)
# --- SOSTITUISCI QUESTO BLOCCO NELLO SCRIPT ---
make_pdp <- function(m, train_use, v, pred_fun){
  pd <- partial(
    object = m, pred.var = v, train = train_use,
    pred.fun = pred_fun, grid.resolution = 30,
    ice = TRUE, frac.ice = 0.01   # ICE leggere
  )
  
  # alcune versioni di pdp non aggiungono .id: creala
  if (!(".id" %in% names(pd))) pd$.id <- 1L
  
  # calcola la linea media (partial dependence)
  avg <- pd %>%
    dplyr::group_by(x = .data[[v]]) %>%
    dplyr::summarise(yhat = mean(yhat), .groups = "drop")
  
  # limita il numero di segmenti ICE se enorme
  if (nrow(pd) > 100000) pd <- pd[sample(nrow(pd), 100000), ]
  
  g <- ggplot(pd, aes(x = .data[[v]], y = yhat, group = .id)) +
    geom_line(alpha = 0.05) +                      # ICE leggere
    geom_line(data = avg, aes(x = x, y = yhat),    # linea media
              inherit.aes = FALSE, linewidth = 1.1) +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    labs(title = paste("PDP/ICE —", v), x = v, y = "Pr(default)") +
    theme_minimal(base_size = 12)
  
  list(plot = g, pd = pd)
}
# --- FINE BLOCCO SOSTITUITO ---



# 10) Costruisci e salva le figure + ranges
train_use <- train[, unique(c(feat_names, target)), drop=FALSE]
ranges <- list()

for (v in vars) {
  res <- make_pdp(m, train_use, v, pred_fun)
  g   <- res$plot; pd <- res$pd
  out_png <- file.path("outputs/figures", paste0("pdp_", v, "_pretty.png"))
  ggsave(out_png, g, w = 7, h = 4, dpi = 200)
  ranges[[length(ranges)+1]] <- data.frame(
    var = v,
    x_min = min(pd[[v]]), x_max = max(pd[[v]]),
    pd_min = min(pd$yhat), pd_max = max(pd$yhat)
  )
  cat("Saved:", out_png, "\n")
}

readr::write_csv(dplyr::bind_rows(ranges), "outputs/metrics/pdp_ranges.csv")
cat("OK: outputs/figures/pdp_*_pretty.png ; outputs/metrics/pdp_ranges.csv\n")



# === PDP pulite per il report (senza ICE) ===
make_pdp_clean <- function(m, train_use, v, pred_fun){
  pd <- partial(
    object = m, pred.var = v, train = train_use,
    pred.fun = pred_fun, grid.resolution = 60,
    ice = FALSE, plot = FALSE
  )
  # quantili per "rug" orizzontale
  qs <- stats::quantile(train_use[[v]], probs = c(.05,.25,.5,.75,.95), na.rm = TRUE)
  avg <- dplyr::summarise(dplyr::group_by(pd, x = .data[[v]]),
                          yhat = mean(yhat), .groups = "drop")
  g <- ggplot(avg, aes(x = x, y = yhat)) +
    geom_line(linewidth = 1.2) +
    geom_rug(data = data.frame(x = as.numeric(qs)), aes(x = x), sides = "b", inherit.aes = FALSE, alpha = .5) +
    coord_cartesian(ylim = c(0, 1)) +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    labs(title = paste("PDP —", v), x = v, y = "Pr(default)") +
    theme_minimal(base_size = 12)
  g
}

train_use <- train[, unique(c(feat_names, target)), drop=FALSE]

for (v in vars) {
  g_clean <- make_pdp_clean(m, train_use, v, pred_fun)
  fn <- file.path("outputs/figures", paste0("pdp_", v, "_clean.png"))
  ggsave(fn, g_clean, width = 7, height = 4, dpi = 200)
  cat("Saved (clean):", fn, "\n")
  



