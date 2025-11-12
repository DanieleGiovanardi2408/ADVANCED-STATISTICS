# scripts/06_ablation_study.R — base → +FE → +segment (AUC/Brier su val & test)
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(pROC)
})

set.seed(7)
dir.create("outputs/metrics", TRUE, TRUE)

# ---------- helpers ----------
pick_first <- function(...) { x <- c(...); x[file.exists(x)][1] }
find_target <- function(nms) intersect(c("default","y","target","label"), nms)[1]

find_prob_col <- function(df, ycol){
  pref <- c("prob","prob_logit","probability","pred_prob","p","yhat","score",".fitted","pred","prob1")
  cand <- pref[pref %in% names(df)]
  if (length(cand)) return(cand[1])
  numc <- setdiff(names(df)[sapply(df, is.numeric)], ycol)
  in01 <- vapply(numc, function(nm){
    x <- df[[nm]]; q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
  }, logical(1))
  cand2 <- numc[in01]; if (!length(cand2)) stop("Prob column non trovata.")
  cand2[1]
}

brier <- function(p,y) mean((p - y)^2)

# ---------- load base probs (logit) ----------
tr_wp <- pick_first("data/interim/train_with_probs.csv")
va_wp <- pick_first("data/interim/val_with_probs.csv")
te_wp <- pick_first("data/interim/test_with_probs.csv")
stopifnot(!is.na(tr_wp), !is.na(va_wp), !is.na(te_wp))

TrP <- suppressMessages(read_csv(tr_wp, show_col_types = FALSE))
VaP <- suppressMessages(read_csv(va_wp, show_col_types = FALSE))
TeP <- suppressMessages(read_csv(te_wp, show_col_types = FALSE))

ycol <- find_target(names(TrP)); stopifnot(!is.na(ycol))
pcol_tr <- find_prob_col(TrP, ycol)
pcol_va <- find_prob_col(VaP, ycol)
pcol_te <- find_prob_col(TeP, ycol)

y_tr <- as.integer(TrP[[ycol]]); p_tr <- pmin(pmax(as.numeric(TrP[[pcol_tr]]), 1e-8), 1-1e-8)
y_va <- as.integer(VaP[[ycol]]); p_va <- pmin(pmax(as.numeric(VaP[[pcol_va]]), 1e-8), 1-1e-8)
y_te <- as.integer(TeP[[ycol]]); p_te <- pmin(pmax(as.numeric(TeP[[pcol_te]]), 1e-8), 1-1e-8)

# ---------- Base model: y ~ prob ----------
df_tr_base <- tibble(y = y_tr, prob = p_tr)
df_va_base <- tibble(y = y_va, prob = p_va)
df_te_base <- tibble(y = y_te, prob = p_te)

m_base <- glm(y ~ prob, data = df_tr_base, family = binomial())
pr_va_base <- as.numeric(predict(m_base, newdata = df_va_base, type = "response"))
pr_te_base <- as.numeric(predict(m_base, newdata = df_te_base, type = "response"))

auc_va_base <- as.numeric(auc(roc(df_va_base$y, pr_va_base, quiet=TRUE)))
auc_te_base <- as.numeric(auc(roc(df_te_base$y, pr_te_base, quiet=TRUE)))
br_va_base  <- brier(pr_va_base, df_va_base$y)
br_te_base  <- brier(pr_te_base, df_te_base$y)

# ---------- +FE: aggiungi feature numeriche comuni ----------
tr_fx <- pick_first("data/interim/train_features.csv", "data/interim/train.csv")
va_fx <- pick_first("data/interim/val_features.csv",   "data/interim/val.csv")
te_fx <- pick_first("data/interim/test_features.csv",  "data/interim/test.csv")
stopifnot(!is.na(tr_fx), !is.na(va_fx), !is.na(te_fx))

TrX <- suppressMessages(read_csv(tr_fx, show_col_types = FALSE))
VaX <- suppressMessages(read_csv(va_fx, show_col_types = FALSE))
TeX <- suppressMessages(read_csv(te_fx, show_col_types = FALSE))

drop_cols <- unique(c(ycol, "id","ID","customer_id","client_id","segment","segment_named"))
common <- Reduce(intersect, list(names(TrX), names(VaX), names(TeX)))
num_feats <- setdiff(common, drop_cols)
num_feats <- num_feats[sapply(TrX[num_feats], is.numeric)]
use_fe <- length(num_feats) > 0

if (use_fe) {
  df_tr_fe <- tibble(y = y_tr, prob = p_tr) %>% bind_cols(TrX[, num_feats, drop=FALSE])
  df_va_fe <- tibble(y = y_va, prob = p_va) %>% bind_cols(VaX[, num_feats, drop=FALSE])
  df_te_fe <- tibble(y = y_te, prob = p_te) %>% bind_cols(TeX[, num_feats, drop=FALSE])
  
  form_fe <- reformulate(c("prob", num_feats), response = "y")
  m_fe <- glm(form_fe, data = df_tr_fe, family = binomial())
  
  pr_va_fe <- as.numeric(predict(m_fe, newdata = df_va_fe, type = "response"))
  pr_te_fe <- as.numeric(predict(m_fe, newdata = df_te_fe, type = "response"))
  
  auc_va_fe <- as.numeric(auc(roc(df_va_fe$y, pr_va_fe, quiet=TRUE)))
  auc_te_fe <- as.numeric(auc(roc(df_te_fe$y, pr_te_fe, quiet=TRUE)))
  br_va_fe  <- brier(pr_va_fe, df_va_fe$y)
  br_te_fe  <- brier(pr_te_fe, df_te_fe$y)
}

# ---------- +Segment: aggiungi factor(segment) se disponibile ----------
seg_tr <- pick_first("data/interim/train_with_segment_named.csv",
                     "data/interim/train_with_segment.csv",
                     "data/interim/segments_train.csv")
seg_va <- pick_first("data/interim/val_with_segment_named.csv",
                     "data/interim/val_with_segment.csv",
                     "data/interim/segments_val.csv")
seg_te <- pick_first("data/interim/test_with_segment_named.csv",
                     "data/interim/test_with_segment.csv",
                     "data/interim/segments_test.csv")

use_seg <- !any(is.na(c(seg_tr, seg_va, seg_te)))
if (use_seg) {
  TrS <- suppressMessages(read_csv(seg_tr, show_col_types = FALSE))
  VaS <- suppressMessages(read_csv(seg_va, show_col_types = FALSE))
  TeS <- suppressMessages(read_csv(seg_te, show_col_types = FALSE))
  seg_name <- if ("segment_named" %in% names(TrS)) "segment_named" else if ("segment" %in% names(TrS)) "segment" else NA_character_
  if (is.na(seg_name)) use_seg <- FALSE
}

if (use_seg) {
  cl_tr <- factor(TrS[[seg_name]])
  cl_va <- factor(VaS[[seg_name]], levels = levels(cl_tr))
  cl_te <- factor(TeS[[seg_name]], levels = levels(cl_tr))
  
  if (use_fe) {
    df_tr_seg <- tibble(y = y_tr, prob = p_tr) %>% bind_cols(TrX[, num_feats, drop=FALSE]) %>% mutate(cl = cl_tr)
    df_va_seg <- tibble(y = y_va, prob = p_va) %>% bind_cols(VaX[, num_feats, drop=FALSE]) %>% mutate(cl = cl_va)
    df_te_seg <- tibble(y = y_te, prob = p_te) %>% bind_cols(TeX[, num_feats, drop=FALSE]) %>% mutate(cl = cl_te)
    form_seg  <- reformulate(c("prob", num_feats, "cl"), response = "y")
  } else {
    df_tr_seg <- tibble(y = y_tr, prob = p_tr, cl = cl_tr)
    df_va_seg <- tibble(y = y_va, prob = p_va, cl = cl_va)
    df_te_seg <- tibble(y = y_te, prob = p_te, cl = cl_te)
    form_seg  <- y ~ prob + cl
  }
  
  m_seg <- glm(form_seg, data = df_tr_seg, family = binomial())
  pr_va_seg <- as.numeric(predict(m_seg, newdata = df_va_seg, type = "response"))
  pr_te_seg <- as.numeric(predict(m_seg, newdata = df_te_seg, type = "response"))
  
  auc_va_seg <- as.numeric(auc(roc(df_va_seg$y, pr_va_seg, quiet=TRUE)))
  auc_te_seg <- as.numeric(auc(roc(df_te_seg$y, pr_te_seg, quiet=TRUE)))
  br_va_seg  <- brier(pr_va_seg, df_va_seg$y)
  br_te_seg  <- brier(pr_te_seg, df_te_seg$y)
}

# ---------- assemble results ----------
rows <- list(
  tibble(model="Base (y~prob)",
         auc_val=auc_va_base, brier_val=br_va_base,
         auc_test=auc_te_base, brier_test=br_te_base)
)

if (use_fe) rows[[length(rows)+1]] <- tibble(
  model="+FE (y~prob+features)",
  auc_val=auc_va_fe, brier_val=br_va_fe,
  auc_test=auc_te_fe, brier_test=br_te_fe
)

if (use_seg) rows[[length(rows)+1]] <- tibble(
  model=if (use_fe) "+FE+Segment" else "+Segment (y~prob+cl)",
  auc_val=auc_va_seg, brier_val=br_va_seg,
  auc_test=auc_te_seg, brier_test=br_te_seg
)

out <- bind_rows(rows)
readr::write_csv(out, "outputs/metrics/ablation.csv")
print(out)
cat("Saved -> outputs/metrics/ablation.csv\n")
