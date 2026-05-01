import pandas as pd
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist


df_speakers = pd.read_csv('C:\\Users\\aviba\\data\\metadata_parsed.csv')
df_speakers = df_speakers.drop_duplicates(subset=["key"])
print(f"Rows after dedup: {len(df_speakers)}")  # should be 239

data = np.load("C:\\Users\\aviba\\data\\representations_float32.npz")
reps_f64 = {k: data[k].astype(np.float64) for k in data.files}
reps_f32 = {k: data[k].astype(np.float32) for k in data.files}
reps_f16 = {k: data[k].astype(np.float16) for k in data.files}

assert set(df_speakers["key"]) == set(reps_f64.keys()), "mismatch between metadata and npz!"

def quantise_int8(vec: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin, vmax = vec.min(), vec.max()
    scaled = (vec - vmin) / (vmax - vmin)
    quantised = (scaled * 255 - 128).round().astype(np.int8)
    return quantised, vmin, vmax

def dequantise_int8(q: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (q.astype(np.float32) + 128) / 255
    return scaled * (vmax - vmin) + vmin

reps_int8 = {}
for k, vec in reps_f32.items():
    q, vmin, vmax = quantise_int8(vec)
    reps_int8[k] = dequantise_int8(q, vmin, vmax)

def compute_distances(df, reps, precision_name):
    rows = []
    for word, group in df.groupby("word"):
        keys     = group["key"].values
        speakers = group["speaker"].values

        matrix = np.stack([reps[k] for k in keys]).astype(np.float64)
        dists  = cdist(matrix, matrix, metric="cosine")

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                rows.append({
                    "precision": precision_name,
                    "word":      word,
                    "key1":      keys[i],
                    "key2":      keys[j],
                    "speaker1":  speakers[i],
                    "speaker2":  speakers[j],
                    "distance":  dists[i, j],
                    "type":      "intra" if speakers[i] == speakers[j] else "inter",
                })
    return pd.DataFrame(rows)

results = pd.concat([
    compute_distances(df_speakers, reps_f64,  "float64"),
    compute_distances(df_speakers, reps_f32,  "float32"),
    compute_distances(df_speakers, reps_f16,  "float16"),
    compute_distances(df_speakers, reps_int8, "int8"),
])

results.to_csv("C:\\Users\\aviba\\data\\distances.csv", index=False)

summary = results.groupby(["precision", "type"])["distance"].mean().unstack()
summary["ratio"] = summary["inter"] / summary["intra"]
print(f"\nTotal rows: {len(results)}")
print(summary)
