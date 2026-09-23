"""
RAEIPHI — CSI Dataset Exploration
===================================
Loads the generated CSI dataset and produces exploratory
analysis: distributions, correlations, feature behavior
across occupancy classes, and signal pattern visualizations.

Saves all plots to ml/notebooks/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os

# ── Config ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "csi_dataset.csv")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="viridis")


def load_data():
    df = pd.read_csv(DATA_FILE)
    print("=" * 60)
    print("RAEIPHI — CSI Dataset Exploration")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples:  {df.shape[0]:,}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"\nNull values: {df.isnull().sum().sum()}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    return df


def plot_class_distribution(df):
    """Bar chart of occupancy class counts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["occupancy"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
    ax.set_xlabel("Occupancy Count", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution — Occupancy Counts", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, 11))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_amplitude_vs_occupancy(df):
    """Box plot showing how global amplitude stats change with occupancy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = [
        ("amp_mean_global", "Mean Amplitude (dB)"),
        ("amp_std_global", "Amplitude Std Dev"),
        ("amp_range_global", "Amplitude Range"),
    ]

    for ax, (feat, label) in zip(axes, features):
        sns.boxplot(data=df, x="occupancy", y=feat, ax=ax, palette="viridis")
        ax.set_xlabel("Occupancy Count", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label + " by Occupancy", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_amplitude_vs_occupancy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_phase_vs_occupancy(df):
    """Box plot showing phase instability across occupancy levels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    features = [
        ("phase_std_global", "Phase Std Dev (Global)"),
        ("phase_std_max", "Phase Std Dev (Max)"),
    ]

    for ax, (feat, label) in zip(axes, features):
        sns.boxplot(data=df, x="occupancy", y=feat, ax=ax, palette="magma")
        ax.set_xlabel("Occupancy Count", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label + " by Occupancy", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_phase_vs_occupancy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_features(df):
    """Subcarrier correlation and temporal dynamics vs occupancy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = [
        ("subcarrier_corr_mean", "Subcarrier Correlation (Mean)"),
        ("temporal_diff_mean", "Temporal Diff (Mean)"),
        ("temporal_diff_max", "Temporal Diff (Max)"),
    ]

    for ax, (feat, label) in zip(axes, features):
        sns.boxplot(data=df, x="occupancy", y=feat, ax=ax, palette="coolwarm")
        ax.set_xlabel("Occupancy Count", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label + " by Occupancy", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_correlation_temporal_vs_occupancy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_subcarrier_heatmap(df):
    """Heatmap of mean amplitude std across subcarriers for each occupancy level."""
    amp_std_cols = [f"amp_std_sc{i}" for i in range(52)]

    # Average amplitude std per subcarrier per occupancy level
    heatmap_data = df.groupby("occupancy")[amp_std_cols].mean()
    heatmap_data.columns = [f"SC{i}" for i in range(52)]

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean Amplitude Std"})
    ax.set_xlabel("Subcarrier Index", fontsize=12)
    ax.set_ylabel("Occupancy Count", fontsize=12)
    ax.set_title("Signal Disturbance Heatmap — Amplitude Variance per Subcarrier by Occupancy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_subcarrier_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_correlation_matrix(df):
    """Correlation matrix of the global/aggregate features."""
    global_features = [
        "amp_mean_global", "amp_std_global", "amp_range_global",
        "amp_std_max", "amp_std_min", "amp_variance_spread",
        "phase_std_global", "phase_std_max", "phase_std_spread",
        "subcarrier_corr_mean", "subcarrier_corr_std",
        "temporal_diff_mean", "temporal_diff_max", "temporal_diff_std",
        "occupancy"
    ]

    corr = df[global_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix (Global Features + Occupancy)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "06_feature_correlation_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_pairplot_globals(df):
    """Pairplot of key global features colored by occupancy."""
    key_features = [
        "amp_std_global", "amp_range_global",
        "phase_std_global", "subcarrier_corr_mean",
        "temporal_diff_mean", "occupancy"
    ]
    subset = df[key_features].copy()
    # Bin occupancy for clearer color separation
    subset["occ_group"] = pd.cut(subset["occupancy"],
                                  bins=[-1, 0, 2, 5, 10],
                                  labels=["Empty", "1-2", "3-5", "6-10"])

    g = sns.pairplot(subset, hue="occ_group", palette="viridis",
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15},
                     vars=key_features[:-1])
    g.figure.suptitle("Feature Pairplot — Key Global Features by Occupancy Group",
                      y=1.02, fontsize=14, fontweight="bold")
    path = os.path.join(FIG_DIR, "07_pairplot_globals.png")
    g.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def print_summary_stats(df):
    """Print key statistical summaries."""
    global_features = [
        "amp_mean_global", "amp_std_global", "amp_range_global",
        "phase_std_global", "subcarrier_corr_mean",
        "temporal_diff_mean", "temporal_diff_max"
    ]

    print("\n" + "=" * 60)
    print("KEY FEATURE STATISTICS BY OCCUPANCY")
    print("=" * 60)

    for feat in global_features:
        print(f"\n── {feat} ──")
        stats = df.groupby("occupancy")[feat].agg(["mean", "std", "min", "max"])
        print(stats.round(4).to_string())

    # Feature-to-label correlations
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH OCCUPANCY (top 15)")
    print("=" * 60)
    correlations = df.corr()["occupancy"].drop("occupancy").abs().sort_values(ascending=False)
    for feat, corr_val in correlations.head(15).items():
        direction = "+" if df.corr()["occupancy"][feat] > 0 else "-"
        print(f"  {direction} {corr_val:.4f}  {feat}")


def main():
    df = load_data()

    print("\nGenerating visualizations...")
    plot_class_distribution(df)
    plot_amplitude_vs_occupancy(df)
    plot_phase_vs_occupancy(df)
    plot_correlation_features(df)
    plot_subcarrier_heatmap(df)
    plot_feature_correlation_matrix(df)
    plot_pairplot_globals(df)

    print_summary_stats(df)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
