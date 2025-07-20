import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
# Intra-person (flat list of all pairwise segment distances across persons)
INTRA_PKL = ""

# Inter-person (dictionary: person_id -> list of HD to all *other* persons)
# This must be the same structure you already plotted.
INTER_PKL = ""

ARCH_NAME = "CNN+Attention"   # change per architecture if you want
SAVE_DIR = Path("figures_combined")
SAVE_DIR.mkdir(exist_ok=True)


# LOAD INTRA
with open(INTRA_PKL, "rb") as f:
    intra_hd = np.asarray(pickle.load(f), dtype=float)

raw_mean = intra_hd.mean()
raw_std = intra_hd.std(ddof=0)
print(f"[RAW] Intra-person HD: mean={raw_mean:.2f} bits, std={raw_std:.2f}, n={len(intra_hd)}")

# TRIM USING IQR
q1, q3 = np.percentile(intra_hd, [25, 75])
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
trim_mask = (intra_hd >= lower_fence) & (intra_hd <= upper_fence)
intra_trim = intra_hd[trim_mask]
trim_mean = intra_trim.mean()
trim_std  = intra_trim.std(ddof=0)
print(f"[TRIMMED] IQR fences [{lower_fence:.2f}, {upper_fence:.2f}] "
      f"kept {trim_mask.sum()}/{len(intra_hd)} ({trim_mask.sum()/len(intra_hd)*100:.1f}%). "
      f"mean={trim_mean:.2f}, std={trim_std:.2f}")

# ----- WINSORIZE (1%, 99%) -----
p_lo, p_hi = np.percentile(intra_hd, [1, 99])
intra_wins = intra_hd.copy()
intra_wins[intra_wins < p_lo] = p_lo
intra_wins[intra_wins > p_hi] = p_hi
wins_mean = intra_wins.mean()
wins_std  = intra_wins.std(ddof=0)
print(f"[WINSORIZED] caps ({p_lo:.2f}, {p_hi:.2f}) mean={wins_mean:.2f}, std={wins_std:.2f}")

# ----- SQRT TRANSFORM (just for visualization) -----
intra_sqrt = np.sqrt(intra_hd)
sqrt_mean = intra_sqrt.mean()
sqrt_std  = intra_sqrt.std(ddof=0)

# -----------------------------
# PLOT INTRA (FOUR VARIANTS)
def intra_boxplot(data, title, subtitle, ylabel="Hamming Distance (bits)", fname="tmp.png",
                  showfliers=True, ylim=None, label="Intra-person"):
    plt.figure(figsize=(6,6))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showfliers=showfliers,
        boxprops=dict(facecolor='skyblue', edgecolor='navy'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='navy'),
        capprops=dict(color='navy'),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='white',
                        markeredgecolor='black', alpha=0.7)
    )
    plt.xticks([1], [label], fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title + "\n" + subtitle, fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / fname, dpi=300)
    plt.close()

# 1. RAW (cap y-axis just above upper_fence so the flier cloud is still shown but not dominating)
intra_boxplot(
    intra_hd,
    f"Intra-person Hamming Distances ({ARCH_NAME})",
    f"RAW Mean={raw_mean:.2f}, SD={raw_std:.2f}, n={len(intra_hd)}",
    ylim=(-2, min(upper_fence + 12, intra_hd.max()+5)),
    fname="intra_raw.png"
)

# 2. TRIMMED (plot after removing statistical outliers)
intra_boxplot(
    intra_trim,
    f"Intra-person Hamming Distances ({ARCH_NAME})",
    f"TRIMMED Mean={trim_mean:.2f}, SD={trim_std:.2f}, kept={len(intra_trim)}",
    ylim=(-2, intra_trim.max()+5),
    fname="intra_trimmed.png"
)

# 3. WINSORIZED
intra_boxplot(
    intra_wins,
    f"Intra-person Hamming Distances ({ARCH_NAME})",
    f"WINS Mean={wins_mean:.2f}, SD={wins_std:.2f}",
    ylim=(-2, p_hi + 8),
    fname="intra_winsorized.png"
)

# 4. SQRT TRANSFORM
intra_boxplot(
    intra_sqrt,
    f"SQRT Transform Intra-person HD ({ARCH_NAME})",
    f"Mean={sqrt_mean:.2f}, SD={sqrt_std:.2f}",
    ylabel="sqrt(Hamming Distance)",
    fname="intra_sqrt.png"
)

#SIMPLE SINGLE INTRA PLOT
plt.figure(figsize=(8,6))
bp_intra = plt.boxplot(intra_hd, patch_artist=True, showfliers=True)
for patch in bp_intra['boxes']:
    patch.set_facecolor('skyblue')
plt.xlabel("Intra-person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Intra-person Hamming Distances", fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "intra_original_style.png", dpi=300)
plt.close()

# LOAD INTER
with open(INTER_PKL, "rb") as f:
    person_dists = pickle.load(f)

# Expect person_dists: dict {person_id: [distances to others]}
inter_data = [person_dists[p] for p in sorted(person_dists.keys())]

# Overall stats
all_inter = [d for lst in inter_data for d in lst]
overall_inter_mean = np.mean(all_inter)
overall_inter_std = np.std(all_inter)
print("Inter-person Hamming distances: mean = {:.2f} bits, std = {:.2f} bits".format(
    overall_inter_mean, overall_inter_std))

# Group split
group1 = inter_data[:44]   # persons 1-44
group2 = inter_data[44:]   # persons 45-89

# Inter-person Group 1 (unchanged)
plt.figure(figsize=(12,6))
bp1 = plt.boxplot(group1, patch_artist=True, showfliers=True)
colors1 = plt.cm.hsv(np.linspace(0, 1, len(group1)))
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
plt.xlabel("Person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Inter-person Hamming Distances for Persons 1–44", fontsize=20)
plt.xticks(range(1, 45), [str(p) for p in range(1, 45)], rotation=90, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "inter_1_44.png", dpi=300)
plt.close()

# Inter-person Group 2 (unchanged)
plt.figure(figsize=(12,6))
bp2 = plt.boxplot(group2, patch_artist=True, showfliers=True)
colors2 = plt.cm.hsv(np.linspace(0, 1, len(group2)))
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
plt.xlabel("Person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Inter-person Hamming Distances for Persons 45–89", fontsize=20)
plt.xticks(range(1, len(group2)+1), [str(p) for p in range(45, 90)], rotation=90, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "inter_45_89.png", dpi=300)
plt.close()

print(f"All figures saved to: {SAVE_DIR.resolve()}")
