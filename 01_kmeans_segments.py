
# run: python "src/python/01_kmeans_segments.py"
import os, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
train = pd.read_csv("data/interim/train_features.csv")
val   = pd.read_csv("data/interim/val_features.csv")
test  = pd.read_csv("data/interim/test_features.csv")

feat = ["max_dpd","cnt_dpd","util_last","pay_ratio","bill_trend",
        "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
        "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

Xtr = train[feat].fillna(0).values
Xva = val[feat].fillna(0).values
Xte = test[feat].fillna(0).values

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr); Xva_s = scaler.transform(Xva); Xte_s = scaler.transform(Xte)

k = 5
kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(Xtr_s)
print("Silhouette (train):", round(silhouette_score(Xtr_s, kmeans.labels_), 3))

train["segment"] = kmeans.labels_
val["segment"]   = kmeans.predict(Xva_s)
test["segment"]  = kmeans.predict(Xte_s)

os.makedirs("data/interim", exist_ok=True)
train[["row_id","segment"]].to_csv("data/interim/segments_train.csv", index=False)
val[["row_id","segment"]].to_csv("data/interim/segments_val.csv", index=False)
test[["row_id","segment"]].to_csv("data/interim/segments_test.csv", index=False)

os.makedirs("outputs/model_artifacts", exist_ok=True)
joblib.dump(scaler, "outputs/model_artifacts/scaler_kmeans.pkl")
joblib.dump(kmeans, "outputs/model_artifacts/kmeans.pkl")
print("OK: segments saved in data/interim/")

