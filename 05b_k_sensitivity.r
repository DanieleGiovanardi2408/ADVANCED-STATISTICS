# scripts/05b_k_sensitivity.R — sensibilità a k dei cluster (silhouette + ΔAUC val)
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(pROC); library(cluster)
})

set.seed(123)
dir.create("outputs/metrics", TRUE, TRUE)
dir.create("outputs/figures", TRUE, TRUE)

# ---------------- helpers ----------------
pick_file <- function(...) { x <- c(...); x[file.exists(x)][1] }

find_target <- function(nms) {
  cands <- c("default","y","target","label")
  out <- intersect(cands, nms)[1]
  if (is.na(out)) stop("Target non trovato (default/y/target/label).")
  out
}

find_prob_col <- function(df, ycol){
  preferred <- c("prob","prob_logit","probability","pred_prob","p","yhat","score",".fitted","pred","prob1")
  cand <- preferred[preferred %in% names(df)]
  if (length(cand)) return(cand[1])
  numc <- setdiff(names(df)[sapply(df, is.numeric)], ycol)
  if (!length(numc)) return(NA_character_)
  in01 <- vapply(numc, function(nm){
    x <- df[[nm]]; q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
  }, logical(1))
  cand2 <- numc[in01]
  if (!length(cand2)) return(NA_character_)
  aucs <- sapply(cand2, function(nm) suppressWarnings(as.numeric(auc(roc(df[[ycol]], df[[nm]], quiet=TRUE)))))
  cand2[which.max(aucs)]
}

scale_cols <- function(M) {
  numi <- sapply(M, is.numeric)
  M[numi] <- lapply(M[numi], function(v){
    sdv <- sd(v, na.rm=TRUE); mv <- mean(v, na.rm=TRUE)
    if (!is.finite(sdv) || sdv == 0) return(rep(0, length(v)))
    (v - mv)/sdv
  })
  M
}

nearest_centroid <- function(X, centers){
  Xmat <- as.matrix(X)
  d <- as.matrix(dist(rbind(centers, Xmat)))[seq_len(nrow(centers)), (nrow(centers)+1):(nrow(centers)+nrow(Xmat))]
  max.col(-t(d))
}

brier <- function(p,y) mean((p - y)^2)

# ---------------- load data ----------------
train_feat_path <- pick_file("data/interim/train_features.csv", "data/interim/train.csv")
val_feat_path   <- pick_file("data/interim/val_features.csv",   "data/interim/val.csv")
if (any(is.na(c(train_feat_path, val_feat_path)))) {
  stop("Mancano train/val features (data/interim/train_features.csv, val_features.csv).")
}
TrX <- suppressMessages(read_csv(train_feat_path, show_col_types = FALSE))
VaX <- suppressMessages(read_csv(val_feat_path,   show_col_types = FALSE))

# with_probs per PD/logit
train_wp <- pick_file("data/interim/train_with_probs.csv")
val_wp   <- pick_file("data/interim/val_with_probs.csv")
if (any(is.na(c(train_wp, val_wp)))) {
  stop("Mancano train_with_probs.csv e/o val_with_probs.csv in data/interim/ (servono le PD del logit).")
}
TrP <- suppressMessages(read_csv(train_wp, show_col_types = FALSE))
VaP <- suppressMessages(read_csv(val_wp,   show_col_types = FALSE))

if (nrow(TrX) != nrow(TrP) || nrow(VaX) != nrow(VaP)) {
  stop("Riga mismatch tra *_features.csv e *_with_probs.csv (train/val). Devono avere stesso nrow e ordinamento.")
}

# target e prob
ycol <- find_target(names(TrP))
pcol_train <- find_prob_col(TrP, ycol)
pcol_val   <- find_prob_col(VaP, ycol)
if (is.na(pcol_train) || is.na(pcol_val)) stop("Non trovo colonna delle probabilità (logit) in train/val with_probs.")
y_tr <- as.integer(TrP[[ycol]]); p_tr <- pmin(pmax(TrP[[pcol_train]], 1e-6), 1-1e-6)
y_va <- as.integer(VaP[[ycol]]); p_va <- pmin(pmax(VaP[[pcol_val]],   1e-6), 1-1e-6)

# feature numeriche per k-means
drop_cols <- c(ycol, "id","ID","customer_id","client_id","segment","segment_named")
cand_feats <- setdiff(intersect(names(TrX), names(VaX)), drop_cols)
num_feats  <- cand_feats[sapply(TrX[cand_feats], is.numeric)]
if (!length(num_feats)) stop("Nessuna feature numerica comune per k-means.")

TrN <- scale_cols(TrX[, num_feats, drop=FALSE])
VaN <- scale_cols(VaX[, num_feats, drop=FALSE])

# ---------------- loop su k ----------------
K <- 3:8
res <- list()

for (k in K) {
  km <- kmeans(TrN, centers = k, nstart = 20, iter.max = 100)
  sil <- tryCatch({
    mean(silhouette(km$cluster, dist(TrN))[, "sil_width"])
  }, error = function(e) NA_real_)
  
  centers <- km$centers
  cl_va <- nearest_centroid(VaN, centers)
  
  df_tr <- tibble(y = y_tr, prob = p_tr, cl = factor(km$cluster))
  df_va <- tibble(y = y_va, prob = p_va, cl = factor(cl_va, levels = levels(df_tr$cl)))
  
  m_base <- glm(y ~ prob,      data = df_tr, family = binomial())
  m_ext  <- glm(y ~ prob + cl, data = df_tr, family = binomial())
  
  pv_base <- pmin(pmax(as.numeric(predict(m_base, newdata = df_va, type = "response")), 1e-6), 1-1e-6)
  pv_ext  <- pmin(pmax(as.numeric(predict(m_ext,  newdata = df_va, type = "response")),  1e-6), 1-1e-6)
  
  auc_b <- as.numeric(auc(roc(df_va$y, pv_base, quiet = TRUE)))
  auc_e <- as.numeric(auc(roc(df_va$y, pv_ext,  quiet = TRUE)))
  
  res[[length(res)+1]] <- tibble(
    k = k,
    silhouette = sil,
    auc_base_val = auc_b,
    auc_plus_seg_val = auc_e,
    delta_auc = auc_e - auc_b
  )
  cat(sprintf("k=%d | silhouette=%.3f | AUC_base=%.3f | AUC_plus=%.3f | Δ=%.3f\n",
              k, sil, auc_b, auc_e, auc_e - auc_b))
}

out <- bind_rows(res)
write_csv(out, "outputs/metrics/k_sensitivity.csv")

g <- ggplot(out, aes(x = k)) +
  geom_line(aes(y = silhouette), linewidth = 1) +
  geom_point(aes(y = silhouette)) +
  geom_line(aes(y = delta_auc), linewidth = 1, linetype = 2) +
  geom_point(aes(y = delta_auc), shape = 17) +
  scale_x_continuous(breaks = K) +
  labs(title = "Sensibilità a k: silhouette (—) e ΔAUC val (··)",
       x = "k (numero di cluster)", y = "Valore") +
  theme_minimal(base_size = 12)
ggsave("outputs/figures/k_sensitivity.png", g, width = 7, height = 4.5, dpi = 200)

cat("Saved: outputs/metrics/k_sensitivity.csv ; outputs/figures/k_sensitivity.png\n")
