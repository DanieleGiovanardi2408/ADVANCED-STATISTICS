# run: python "src/python/02_kmeans_select_and_assign.py"
import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
K_MIN, K_MAX = 3, 9               # proveremo k = 3..9
FIGDIR = "reports/figures"
# ---------------------------------------

# 1) leggi gli split con le feature create in R
train = pd.read_csv("data/interim/train_features.csv")
val   = pd.read_csv("data/interim/val_features.csv")
test  = pd.read_csv("data/interim/test_features.csv")

# 2) seleziona feature numeriche comportamentali (NO target)
feat = [
  "max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend",
  "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
  "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
]

Xtr = train[feat].fillna(0).values
Xva = val[feat].fillna(0).values
Xte = test[feat].fillna(0).values

# 3) standardizza SOLO su train
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

# 4) grid search su k con silhouette + elbow (WCSS/inertia)
Ks = list(range(K_MIN, K_MAX + 1))

sil_vals = []          # <-- assicurati che siano liste vuote
wcss_vals = []         # <--

for k in Ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xtr_s)
    sil = silhouette_score(Xtr_s, km.labels_)
    sil_vals.append(sil)
    wcss_vals.append(km.inertia_)
    print(f"k={k:>2}  silhouette={sil:.3f}  WCSS={km.inertia_:.1f}")

# Converti in array 1D e controlla lunghezze
Ks_arr    = np.asarray(Ks).reshape(-1)
sil_arr   = np.asarray(sil_vals, dtype=float).reshape(-1)
wcss_arr  = np.asarray(wcss_vals, dtype=float).reshape(-1)
assert len(Ks_arr) == len(sil_arr) == len(wcss_arr), \
    f"Len mismatch: Ks={len(Ks_arr)} sil={len(sil_arr)} wcss={len(wcss_arr)}"

# 5) salva figure (reports/figures)
os.makedirs(FIGDIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # backend non interattivo
import matplotlib.pyplot as plt

plt.figure()
plt.plot(Ks_arr, wcss_arr, marker="o")
plt.title("Elbow plot (WCSS) — train")
plt.xlabel("k"); plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "elbow_wcss.png"), dpi=150)
plt.close()

plt.figure()
plt.bar(Ks_arr, sil_arr)
plt.title("Silhouette score by k — train")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "silhouette_by_k.png"), dpi=150)
plt.close()

# 6) scegli automaticamente k: massimizza silhouette; tie-break su WCSS
best_idx = int(np.argmax(sil_arr))
best_sil = sil_arr[best_idx]
candidates = [i for i, s in enumerate(sil_arr) if abs(s - best_sil) < 1e-6]
if len(candidates) > 1:
    best_idx = min(candidates, key=lambda i: wcss_arr[i])

k_star = int(Ks_arr[best_idx])
print(f"\nChosen k = {k_star}  (silhouette={sil_arr[best_idx]:.3f})")

# 7) rifit + assegnazione
km_final = KMeans(n_clusters=k_star, random_state=RANDOM_STATE, n_init=10).fit(Xtr_s)
train["segment"] = km_final.labels_
val["segment"]   = km_final.predict(Xva_s)
test["segment"]  = km_final.predict(Xte_s)

# 8) salvataggi
os.makedirs("data/interim", exist_ok=True)
train[["row_id","segment"]].to_csv("data/interim/segments_train.csv", index=False)
val[["row_id","segment"]].to_csv("data/interim/segments_val.csv", index=False)
test[["row_id","segment"]].to_csv("data/interim/segments_test.csv", index=False)

with open("outputs/kmeans_selection_summary.json", "w") as f:
    json.dump({
        "Ks": Ks,
        "silhouette": [float(x) for x in sil_arr],
        "wcss": [float(x) for x in wcss_arr],
        "k_star": k_star,
        "silhouette_k_star": float(sil_arr[best_idx])
    }, f, indent=2)

print("OK: segments saved in data/interim/")
print("OK: figures in reports/figures/  & summary in outputs/kmeans_selection_summary.json")
