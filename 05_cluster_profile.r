library(dplyr); library(readr); library(tidyr); library(ggplot2)
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

train <- read_csv("data/interim/train_features.csv") |>
  left_join(read_csv("data/interim/segments_train.csv"), by = "row_id")

vars <- c("max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend",
          paste0("BILL_AMT",1:6), paste0("PAY_AMT",1:6))

# Tabella profili (medie)
prof <- train |>
  group_by(segment) |>
  summarise(across(all_of(vars), ~mean(.x, na.rm=TRUE)), n = dplyr::n())
readr::write_csv(prof, "reports/cluster_profiles_train.csv")
print(prof)

# Heatmap su z-score (più leggibile)
z <- function(x) (x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)
zdf <- train |>
  mutate(across(all_of(vars), z)) |>
  group_by(segment) |>
  summarise(across(all_of(vars), ~mean(.x, na.rm=TRUE))) |>
  pivot_longer(-segment, names_to="variable", values_to="zmean")

p <- ggplot(zdf, aes(x=variable, y=factor(segment), fill=zmean)) +
  geom_tile() + coord_fixed() +
  scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0) +
  labs(title="Cluster profile (z-scores)", x=NULL, y="segment") +
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
ggsave("reports/figures/cluster_profile_heatmap.png", p, width=10, height=4, dpi=150)
