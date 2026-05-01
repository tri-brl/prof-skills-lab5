
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("C:\\Users\\aviba\\data\\distances.csv")

precision_order = ["float64", "float32", "float16", "int8"]

# ── Plot 1: KDE distribution of distances per precision ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, prec in zip(axes, precision_order):
    subset = df[df["precision"] == prec]
    for dtype, color in [("inter", "#4C72B0"), ("intra", "#DD8452")]:
        d = subset[subset["type"] == dtype]["distance"]
        d.plot.kde(ax=ax, label=dtype, color=color, linewidth=2)
    ax.set_title(f"precision = {prec}")
    ax.set_xlabel("cosine distance")
    ax.set_xlim(-0.1, 1.2)
    ax.legend()

axes[0].set_ylabel("density")
fig.suptitle("Distribution of intra- and inter-speaker cosine distances by precision", y=1.02)
plt.tight_layout()
plt.savefig("plots/kde_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plots/kde_distributions.png")

# ── Plot 2: Mean intra vs inter distance across precisions ────────────────────
summary = df.groupby(["precision", "type"])["distance"].mean().reset_index()
summary["precision"] = pd.Categorical(summary["precision"], categories=precision_order, ordered=True)
summary = summary.sort_values("precision")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=summary,
    x="precision", y="distance", hue="type",
    palette={"inter": "#4C72B0", "intra": "#DD8452"},
    ax=ax
)
ax.set_title("Mean intra- vs inter-speaker cosine distance by precision")
ax.set_xlabel("precision")
ax.set_ylabel("mean cosine distance")
ax.legend(title="type")

# annotate bars with values
for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("plots/mean_distances_barplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plots/mean_distances_barplot.png")

# ── Plot 3: Ratio (inter/intra) across precisions ─────────────────────────────
ratio = summary.pivot(index="precision", columns="type", values="distance")
ratio["ratio"] = ratio["inter"] / ratio["intra"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(precision_order, ratio.loc[precision_order, "ratio"],
        marker="o", linewidth=2, color="#4C72B0")
ax.axhline(ratio["ratio"].iloc[0], linestyle="--", color="gray", alpha=0.5, label="float64 baseline")
ax.set_title("Inter/intra distance ratio by precision")
ax.set_xlabel("precision")
ax.set_ylabel("ratio (inter / intra)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/ratio_by_precision.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plots/ratio_by_precision.png")