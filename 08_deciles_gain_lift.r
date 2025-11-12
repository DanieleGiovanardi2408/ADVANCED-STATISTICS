# --- Decili, Gain e Lift per qualsiasi file *_with_probs*.csv ---
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(scales)
})

dir.create("outputs/metrics", TRUE, TRUE)
dir.create("outputs/figures", TRUE, TRUE)

# Scegli automaticamente una o più sorgenti
cands <- c("data/interim/test_with_probs.csv",       # logit
           "data/interim/test_with_probs_rf.csv")    # rf
cands <- cands[file.exists(cands)]
stopifnot(length(cands) >= 1)

find_target <- function(nms) intersect(c("default","y","target","label"), nms)[1]
find_prob   <- function(df) {
  pref <- c("prob","prob_logit","prob_reg","pred_prob","p","yhat","score",".fitted","pred","prob_rf")
  px <- pref[pref %in% names(df)]
  if (length(px)) px[1] else NA_character_
}

make_one <- function(path){
  df <- suppressMessages(read_csv(path, show_col_types = FALSE))
  ycol <- find_target(names(df)); stopifnot(!is.na(ycol))
  pcol <- find_prob(df);         stopifnot(!is.na(pcol))
  y <- as.integer(df[[ycol]])
  p <- pmin(pmax(df[[pcol]], 1e-6), 1-1e-6)
  
  # ranking per decili (1 = più rischio)
  ord <- order(-p)
  dec <- cut(seq_along(p), breaks=quantile(seq_along(p), probs=seq(0,1,by=.1)),
             include.lowest=TRUE, labels=FALSE)
  dec <- dec[ord]  # allineo all'ordinamento
  tab <- tibble(row=seq_along(p)[ord], y=y[ord], p=p[ord], dec=dec) |>
    group_by(dec) |>
    summarise(n=n(), defaults=sum(y), mean_p=mean(p), .groups="drop") |>
    arrange(dec)
  
  total_def <- sum(y)
  tab <- tab |>
    mutate(cum_defaults = cumsum(defaults),
           gain = cum_defaults / total_def,
           lift = gain / (dec/10),
           model = gsub("^test_with_probs(_)?", "", basename(path)))
  
  # figure gain & lift
  g_gain <- ggplot(tab, aes(x = dec/10, y = gain)) +
    geom_line(linewidth=1) + geom_point() +
    scale_y_continuous(labels=percent_format(accuracy = 1)) +
    scale_x_continuous(labels=percent_format(accuracy = 1), breaks=seq(.1,1,.1)) +
    labs(title=paste("Gain curve —", basename(path)),
         x="% popolazione (ordinata per PD)", y="% default catturati") +
    theme_minimal(base_size=12)
  
  g_lift <- ggplot(tab, aes(x = dec/10, y = lift)) +
    geom_line(linewidth=1) + geom_point() +
    labs(title=paste("Lift curve —", basename(path)),
         x="% popolazione", y="Lift") +
    theme_minimal(base_size=12)
  
  pf <- paste0("outputs/figures/gain_", tools::file_path_sans_ext(basename(path)), ".png")
  lf <- paste0("outputs/figures/lift_", tools::file_path_sans_ext(basename(path)), ".png")
  ggsave(pf, g_gain, width=6.5, height=4.5, dpi=200)
  ggsave(lf, g_lift, width=6.5, height=4.5, dpi=200)
  
  out <- paste0("outputs/metrics/deciles_", tools::file_path_sans_ext(basename(path)), ".csv")
  write_csv(tab, out)
  cat("OK →", out, "\n")
}

invisible(lapply(cands, make_one))
