# scripts/04_calibration_brier.R
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(scales)
})

dir.create("outputs/figures", TRUE, TRUE)
dir.create("outputs/metrics", TRUE, TRUE)

# --- helper: trova target/prob in modo robusto
find_target <- function(nms) {
  out <- intersect(c("default","y","target","label"), nms)[1]
  if (is.na(out)) stop("Target non trovato (default/y/target/label).")
  out
}
find_prob_col <- function(df, ycol){
  pref <- c("prob","prob_logit","probability","pred_prob","p","yhat","score",".fitted","pred","prob1")
  cand <- pref[pref %in% names(df)]
  if (length(cand)) return(cand[1])
  # fallback: numeriche in [0,1]
  numc <- setdiff(names(df)[sapply(df, is.numeric)], ycol)
  in01 <- vapply(numc, function(nm){
    x <- df[[nm]]; q <- suppressWarnings(quantile(x, c(.01,.99), na.rm=TRUE))
    all(is.finite(q)) && q[1] >= -0.01 && q[2] <= 1.01
  }, logical(1))
  cand2 <- numc[in01]
  if (!length(cand2)) stop("Colonna probabilità non trovata in with_probs.")
  cand2[1]
}

# --- Brier decomposition con bin per rango (evita 'breaks' duplicati)
brier_decomp_rank <- function(y, p, nbins=10){
  stopifnot(length(y) == length(p))
  # decili per rango (ntile) -> no problemi di ties
  bin <- dplyr::ntile(p, nbins)
  df <- dplyr::tibble(y=y, p=p, bin=bin) |>
    dplyr::group_by(bin) |>
    dplyr::summarise(n=dplyr::n(),
                     p_bar=mean(p),
                     o_bar=mean(y),
                     .groups="drop")
  pi <- mean(y)
  reliability <- sum(df$n * (df$p_bar - df$o_bar)^2) / length(y)
  resolution  <- sum(df$n * (df$o_bar - pi)^2) / length(y)
  bs <- mean((p - y)^2)
  uncertainty <- pi*(1-pi)
  list(bs=bs, reliability=reliability, resolution=resolution, uncertainty=uncertainty, df=df)
}

# --- funzione per uno split (val/test)
do_split <- function(split=c("val","test"), nbins=10){
  split <- match.arg(split)
  path  <- file.path("data/interim", paste0(split, "_with_probs.csv"))
  if (!file.exists(path)) stop("Manca: ", path)
  d <- suppressMessages(readr::read_csv(path, show_col_types = FALSE))
  ycol <- find_target(names(d))
  pcol <- find_prob_col(d, ycol)
  y <- as.integer(d[[ycol]])
  p <- pmin(pmax(as.numeric(d[[pcol]]), 1e-8), 1-1e-8)
  
  dec <- brier_decomp_rank(y, p, nbins)
  
  # plot calibrazione
  g <- ggplot(dec$df, aes(x=p_bar, y=o_bar, size=n)) +
    geom_point(alpha=.85) +
    geom_abline(slope=1, intercept=0, linetype=2) +
    labs(title=paste("Calibration plot (", nbins, " bins) — ", split, sep=""),
         x="Predicted PD (bin mean)", y="Observed default rate") +
    scale_size_area(max_size = 10) +
    coord_equal() +
    theme_minimal(base_size = 12)
  
  f <- file.path("outputs/figures", paste0("calibration_10bins_", split, ".png"))
  ggsave(f, g, width=6.5, height=5.2, dpi=200)
  
  # salva anche i bin per trasparenza
  readr::write_csv(dec$df, file.path("outputs/metrics", paste0("calibration_bins_", split, ".csv")))
  
  tibble(split=split,
         n=length(y),
         bs=dec$bs,
         reliability=dec$reliability,
         resolution=dec$resolution,
         uncertainty=dec$uncertainty,
         prob_col=pcol,
         file=basename(path))
}

# --- run per val & test e salvataggi
out_val  <- do_split("val",  nbins=10)
out_test <- do_split("test", nbins=10)

# file sintetici (manteniamo anche quello atteso dal tuo report)
readr::write_csv(bind_rows(out_val, out_test), "outputs/metrics/brier_decomposition_summary.csv")
readr::write_csv(out_test, "outputs/metrics/brier_decomposition_test.csv")

cat("OK: calibration & brier salvati in outputs/figures/ e outputs/metrics/ (summary + test)\n")

