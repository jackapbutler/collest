"""
Quantitative analysis of emergent message passing in NeuralCA.
Generates two figures:
1. activation_decay_comparison.png - T=0 vs T=18 activation by distance
2. activation_rate_of_change.png - Temporal dynamics of activation updates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage
from spatialca import net, map
import pickle
import pandas as pd
import json
import os

# ============================================================
# Publication styling
# ============================================================

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIG_WIDTH = 3.4
FIG_HEIGHT = 2.2

# ============================================================
# Setup
# ============================================================

COLOUR_MAP = map.satellite_colour_map
num_classes = len(map.satellite_colour_map)
data_source = (
    "openearth_map_train_data_rio"  # Must match the data used for training the models
)
CACHE_FILE = f"data/{data_source}_samples.pkl"

model_T0 = net.NeuralCA.load(f"best_chkpt_CNN-{data_source}.pth")
model_T18 = net.NeuralCA.load(f"best_chkpt_NeuralCA-{data_source}.pth")
print(
    f"Loaded T=0 ({model_T0.full_steps} iterations) and T=18 ({model_T18.full_steps} iterations)"
)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cached_samples = pickle.load(f)
    print(f"Loaded {len(cached_samples)} cached samples")
else:
    print(f"Cache not found at {CACHE_FILE}, creating from raw data...")
    with open(f"./data/{data_source}.json", "r") as f:
        raw_data = json.load(f)
    cached_samples = [{"sample": torch.tensor(s[0])} for s in raw_data]
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cached_samples, f)
    print(f"Created and saved {len(cached_samples)} cached samples to {CACHE_FILE}")

# ============================================================
# Helper function
# ============================================================


def get_activation_by_distance(model, samples, num_classes):
    """Compute mean activation magnitude by distance from observed cells."""
    all_data = []
    for cached in samples:
        sample = cached["sample"]
        unseen = (sample[:, :, -1] == 1).numpy()

        input_oh = map.rgb_to_input_tensor(sample, COLOUR_MAP)
        state = input_oh.unsqueeze(0).clone()

        with torch.no_grad():
            for t in range(model.full_steps):
                if t == 0:
                    state = model(
                        logits=state[:, :num_classes],
                        unseen_mask=state[:, num_classes : num_classes + 1],
                        steps=1,
                        return_state=True,
                    )
                else:
                    state = model(current_state=state, steps=1, return_state=True)

        distance = ndimage.distance_transform_edt(unseen)
        activation = state[0].abs().mean(dim=0).numpy()

        for d in range(1, min(int(distance.max()) + 1, 10)):
            mask = (distance >= d) & (distance < d + 1) & unseen
            if mask.sum() >= 5:
                all_data.append({"distance": d, "activation": activation[mask].mean()})

    df = pd.DataFrame(all_data)
    grouped = (
        df.groupby("distance")
        .agg({"activation": ["mean", "std", "count"]})
        .reset_index()
    )
    grouped.columns = ["distance", "mean", "std", "n"]
    grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["n"])
    return grouped


# ============================================================
# FIGURE 1: T=0 vs T=18 activation decay comparison
# ============================================================

print("\nGenerating activation decay comparison...")

data_T0 = get_activation_by_distance(model_T0, cached_samples, num_classes)
data_T18 = get_activation_by_distance(model_T18, cached_samples, num_classes)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# T=0
ax.fill_between(
    data_T0["distance"],
    data_T0["mean"] - data_T0["ci"],
    data_T0["mean"] + data_T0["ci"],
    alpha=0.15,
    color="#D55E00",
)
ax.plot(
    data_T0["distance"],
    data_T0["mean"],
    "s-",
    color="#D55E00",
    linewidth=1.5,
    markersize=4,
    label=r"NeuralCA($T$=0)",
)

# T=18
ax.fill_between(
    data_T18["distance"],
    data_T18["mean"] - data_T18["ci"],
    data_T18["mean"] + data_T18["ci"],
    alpha=0.15,
    color="#0072B2",
)
ax.plot(
    data_T18["distance"],
    data_T18["mean"],
    "o-",
    color="#0072B2",
    linewidth=1.5,
    markersize=4,
    label=r"NeuralCA($T$=18)",
)

ax.set_xlabel("Distance from observed (cells)")
ax.set_ylabel("Activation magnitude ± 95% CI")
ax.legend(frameon=False, loc="upper right")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

plt.tight_layout(pad=0.3)
plt.savefig(
    "paper/assets/dynamics/activation_decay_comparison.png",
    dpi=600,
    bbox_inches="tight",
)
print("Saved: paper/assets/dynamics/activation_decay_comparison.png")
plt.close()

# ============================================================
# FIGURE 2: Rate of change over iterations by distance band
# ============================================================

print("\nGenerating rate of change plot...")

distance_bands = [1, 2, 3, 5, 7, 10]
all_derivatives = []

for sample_idx, cached in enumerate(cached_samples):
    sample = cached["sample"]
    seen = (sample[:, :, -1] == 0).numpy()
    unseen = (sample[:, :, -1] == 1).numpy()
    input_oh = map.rgb_to_input_tensor(sample, COLOUR_MAP)
    distance = ndimage.distance_transform_edt(unseen)

    states_over_time = []
    state = input_oh.unsqueeze(0).clone()

    with torch.no_grad():
        for t in range(model_T18.full_steps):
            if t == 0:
                state = model_T18(
                    logits=state[:, :num_classes],
                    unseen_mask=state[:, num_classes : num_classes + 1],
                    steps=1,
                    return_state=True,
                )
            else:
                state = model_T18(current_state=state, steps=1, return_state=True)
            states_over_time.append(state[0].abs().mean(dim=0).numpy())

    for d in distance_bands:
        mask = (distance >= d) & (distance < d + 1) & unseen
        if mask.sum() < 5:
            continue

        ts = [states_over_time[t][mask].mean() for t in range(len(states_over_time))]
        derivatives = np.diff(ts)

        for iteration, deriv in enumerate(derivatives):
            all_derivatives.append(
                {
                    "distance": d,
                    "iteration": iteration + 1,
                    "derivative": deriv,
                    "sample": sample_idx,
                }
            )

df = pd.DataFrame(all_derivatives)

grouped = (
    df.groupby(["distance", "iteration"])
    .agg({"derivative": ["mean", "std", "count"]})
    .reset_index()
)
grouped.columns = ["distance", "iteration", "mean", "std", "n"]
grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["n"])

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(distance_bands)))

for d, c in zip(distance_bands, colors):
    data = grouped[grouped["distance"] == d]
    if len(data) > 0:
        ax.fill_between(
            data["iteration"],
            data["mean"] - data["ci"],
            data["mean"] + data["ci"],
            alpha=0.15,
            color=c,
        )
        ax.plot(
            data["iteration"],
            data["mean"],
            "o-",
            color=c,
            linewidth=1.5,
            markersize=3,
            label=f"d={d}",
        )

ax.set_xlabel("Iteration")
ax.set_ylabel("Δ Activation ± 95% CI")
ax.legend(frameon=False, loc="lower right", ncol=2, title="Distance")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

plt.tight_layout(pad=0.3)
plt.savefig(
    "paper/assets/dynamics/activation_rate_of_change.png", dpi=600, bbox_inches="tight"
)
print("Saved: paper/assets/dynamics/activation_rate_of_change.png")
plt.close()

# ============================================================
# Summary statistics for paper
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY STATISTICS FOR PAPER")
print("=" * 60)
print("\nFigure 1: Activation Decay Comparison")
print(f"  T=0:  {data_T0['mean'].iloc[0]:.2f} → {data_T0['mean'].iloc[-1]:.2f}")
print(f"  T=18: {data_T18['mean'].iloc[0]:.2f} → {data_T18['mean'].iloc[-1]:.2f}")
print(
    f"  T=18 has {data_T18['mean'].iloc[0] / data_T0['mean'].iloc[0]:.1f}× higher activation at d=1"
)

print("\nFigure 2: Rate of Change")
for d in distance_bands:
    d_data = grouped[grouped["distance"] == d]
    if len(d_data) > 0:
        early = d_data[d_data["iteration"] <= 3]["mean"].mean()
        late = d_data[d_data["iteration"] >= 15]["mean"].mean()
        print(f"  d={d}: early={early:.3f}, late={late:.3f}")
