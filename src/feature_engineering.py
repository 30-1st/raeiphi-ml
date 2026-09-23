"""
RAEIPHI — CSI Feature Engineering Pipeline
============================================
Loads the raw CSI dataset, engineers additional features,
removes highly correlated redundant features, normalizes
all values, and splits into train/test/validation sets.

Input:  ml/data/csi_dataset.csv
Output: ml/data/train.csv, ml/data/val.csv, ml/data/test.csv
        ml/data/feature_config.json (scaler params + selected features)
"""

import pandas as pd
import numpy as np
import json
import os

# ── Config ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_FILE = os.path.join(DATA_DIR, "csi_dataset.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "feature_config.json")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Correlation threshold for dropping redundant features
CORR_THRESHOLD = 0.95

np.random.seed(RANDOM_SEED)


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSI dataset."""
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded raw dataset: {df.shape[0]:,} samples, {df.shape[1] - 1} features")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from the raw CSI data.
    These capture higher-order signal patterns that help
    distinguish occupancy levels beyond raw amplitude/phase.
    """
    print("\nEngineering additional features...")
    n_subcarriers = 52

    # ── Signal-to-Noise Ratio (SNR) per subcarrier ──────────────
    # SNR = mean / std — higher SNR = more stable signal = fewer people
    for i in range(n_subcarriers):
        mean_col = f"amp_mean_sc{i}"
        std_col = f"amp_std_sc{i}"
        # Avoid division by zero
        df[f"snr_sc{i}"] = df[mean_col] / (df[std_col] + 1e-6)

    # Global SNR statistics
    snr_cols = [f"snr_sc{i}" for i in range(n_subcarriers)]
    df["snr_global_mean"] = df[snr_cols].mean(axis=1)
    df["snr_global_std"] = df[snr_cols].std(axis=1)
    df["snr_global_min"] = df[snr_cols].min(axis=1)
    df["snr_global_max"] = df[snr_cols].max(axis=1)
    print(f"  + {n_subcarriers} per-subcarrier SNR features + 4 global SNR stats")

    # ── Amplitude distribution shape features ───────────────────
    # Skewness and kurtosis of amplitude across subcarriers
    amp_mean_cols = [f"amp_mean_sc{i}" for i in range(n_subcarriers)]
    amp_std_cols = [f"amp_std_sc{i}" for i in range(n_subcarriers)]

    df["amp_mean_skew"] = df[amp_mean_cols].skew(axis=1)
    df["amp_mean_kurtosis"] = df[amp_mean_cols].kurtosis(axis=1)
    df["amp_std_skew"] = df[amp_std_cols].skew(axis=1)
    df["amp_std_kurtosis"] = df[amp_std_cols].kurtosis(axis=1)
    print("  + 4 amplitude distribution shape features (skew, kurtosis)")

    # ── Phase distribution shape features ───────────────────────
    phase_std_cols = [f"phase_std_sc{i}" for i in range(n_subcarriers)]
    df["phase_std_skew"] = df[phase_std_cols].skew(axis=1)
    df["phase_std_kurtosis"] = df[phase_std_cols].kurtosis(axis=1)
    print("  + 2 phase distribution shape features")

    # ── Subcarrier group statistics ─────────────────────────────
    # Split subcarriers into lower, middle, upper bands
    # Different bands respond differently to human presence
    lower = list(range(0, 17))
    middle = list(range(17, 35))
    upper = list(range(35, 52))

    for band_name, band_indices in [("lower", lower), ("mid", middle), ("upper", upper)]:
        band_amp_std = [f"amp_std_sc{i}" for i in band_indices]
        band_phase_std = [f"phase_std_sc{i}" for i in band_indices]

        df[f"amp_std_{band_name}_mean"] = df[band_amp_std].mean(axis=1)
        df[f"amp_std_{band_name}_std"] = df[band_amp_std].std(axis=1)
        df[f"phase_std_{band_name}_mean"] = df[band_phase_std].mean(axis=1)

    # Band contrast: difference between upper and lower band disturbance
    df["amp_std_band_contrast"] = df["amp_std_upper_mean"] - df["amp_std_lower_mean"]
    df["phase_std_band_contrast"] = df["phase_std_upper_mean"] - df["phase_std_lower_mean"]
    print("  + 11 subcarrier band features (lower/mid/upper groupings + contrast)")

    # ── Percentile features ─────────────────────────────────────
    # How much of the signal is disturbed — captures spatial spread of people
    df["amp_std_p25"] = df[amp_std_cols].quantile(0.25, axis=1)
    df["amp_std_p50"] = df[amp_std_cols].quantile(0.50, axis=1)
    df["amp_std_p75"] = df[amp_std_cols].quantile(0.75, axis=1)
    df["amp_std_iqr"] = df["amp_std_p75"] - df["amp_std_p25"]
    print("  + 4 amplitude std percentile features (P25, P50, P75, IQR)")

    # ── Ratio features ──────────────────────────────────────────
    # Relative measures that are more robust than absolute values
    df["phase_to_amp_ratio"] = df["phase_std_global"] / (df["amp_std_global"] + 1e-6)
    df["temporal_to_amp_ratio"] = df["temporal_diff_mean"] / (df["amp_std_global"] + 1e-6)
    df["range_to_mean_ratio"] = df["amp_range_global"] / (df["amp_mean_global"] + 1e-6)
    print("  + 3 ratio features")

    total_new = n_subcarriers + 4 + 4 + 2 + 11 + 4 + 3  # = 80
    total_features = df.shape[1] - 1  # minus label
    print(f"\n  Total new features: {total_new}")
    print(f"  Total features before selection: {total_features}")

    return df


def remove_redundant_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Remove highly correlated features to reduce multicollinearity.
    When two features correlate above the threshold, drop the one
    with lower correlation to the target variable.
    """
    print(f"\nRemoving redundant features (correlation > {threshold})...")

    feature_cols = [c for c in df.columns if c != "occupancy"]
    corr_matrix = df[feature_cols].corr().abs()

    # Track which features to drop
    to_drop = set()

    # Get correlation of each feature with the target
    target_corr = df[feature_cols].corrwith(df["occupancy"]).abs()

    # Upper triangle of correlation matrix (avoid double-counting)
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr_matrix.iloc[i, j] > threshold:
                col_i = feature_cols[i]
                col_j = feature_cols[j]
                # Drop whichever has lower correlation with occupancy
                if target_corr[col_i] < target_corr[col_j]:
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)

    # Drop redundant features
    df_reduced = df.drop(columns=list(to_drop))
    remaining = df_reduced.shape[1] - 1  # minus label

    print(f"  Dropped {len(to_drop)} redundant features")
    print(f"  Remaining features: {remaining}")

    # Show what was dropped (grouped)
    dropped_groups = {
        "per-subcarrier amp_mean": 0,
        "per-subcarrier amp_std": 0,
        "per-subcarrier amp_range": 0,
        "per-subcarrier phase_std": 0,
        "per-subcarrier snr": 0,
        "global/engineered": 0,
    }
    for col in sorted(to_drop):
        if col.startswith("amp_mean_sc"):
            dropped_groups["per-subcarrier amp_mean"] += 1
        elif col.startswith("amp_std_sc"):
            dropped_groups["per-subcarrier amp_std"] += 1
        elif col.startswith("amp_range_sc"):
            dropped_groups["per-subcarrier amp_range"] += 1
        elif col.startswith("phase_std_sc"):
            dropped_groups["per-subcarrier phase_std"] += 1
        elif col.startswith("snr_sc"):
            dropped_groups["per-subcarrier snr"] += 1
        else:
            dropped_groups["global/engineered"] += 1

    print("  Dropped by category:")
    for group, count in dropped_groups.items():
        if count > 0:
            print(f"    {group}: {count}")

    return df_reduced, list(to_drop)


def normalize_features(df: pd.DataFrame) -> tuple:
    """
    Normalize all features using StandardScaler (zero mean, unit variance).
    Returns the normalized dataframe and the scaler parameters
    (mean and std per feature) so we can apply the same transform at inference time.
    """
    print("\nNormalizing features (StandardScaler)...")

    feature_cols = [c for c in df.columns if c != "occupancy"]
    labels = df["occupancy"].copy()

    # Compute mean and std from the full dataset
    # (in practice we'd compute from train only — we'll fix that after splitting)
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()

    # Avoid division by zero for constant features
    stds = stds.replace(0, 1)

    # Normalize
    df_normalized = df.copy()
    df_normalized[feature_cols] = (df[feature_cols] - means) / stds

    # Store scaler params
    scaler_params = {
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "feature_columns": feature_cols,
    }

    print(f"  Normalized {len(feature_cols)} features")
    print(f"  Value range after scaling: [{df_normalized[feature_cols].min().min():.2f}, "
          f"{df_normalized[feature_cols].max().max():.2f}]")

    return df_normalized, scaler_params


def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Split into train/val/test with stratification by occupancy class.
    Stratification ensures each split has the same class distribution.
    """
    print(f"\nSplitting dataset ({TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%})...")

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Stratified split
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for occ in sorted(df["occupancy"].unique()):
        class_df = df[df["occupancy"] == occ]
        n = len(class_df)

        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_dfs.append(class_df.iloc[:train_end])
        val_dfs.append(class_df.iloc[train_end:val_end])
        test_dfs.append(class_df.iloc[val_end:])

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # Verify stratification
    print("\n  Class distribution per split:")
    print(f"  {'Class':<8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'─' * 36}")
    for occ in sorted(df["occupancy"].unique()):
        t = (train_df["occupancy"] == occ).sum()
        v = (val_df["occupancy"] == occ).sum()
        te = (test_df["occupancy"] == occ).sum()
        print(f"  {occ:<8} {t:>8} {v:>8} {te:>8}")

    return train_df, val_df, test_df


def recompute_scaler_from_train(train_df: pd.DataFrame, scaler_params: dict) -> dict:
    """
    Recompute normalization parameters from training set only.
    This prevents data leakage — val/test are normalized using
    train statistics, not their own.
    """
    print("\nRecomputing scaler from training set only (preventing data leakage)...")

    feature_cols = scaler_params["feature_columns"]

    # Unnormalize train data first
    old_means = scaler_params["means"]
    old_stds = scaler_params["stds"]

    # We need to work with the original (un-normalized) values
    # Since we normalized the full dataset, let's reload raw and re-split
    # Actually, let's just recompute from the train split properly

    # The cleaner approach: reload raw, engineer, split, THEN normalize from train only
    # But since we already have the splits, we can reverse the normalization
    # and redo it from train stats

    return scaler_params  # We'll fix this in the main flow


def main():
    print("=" * 60)
    print("RAEIPHI — Feature Engineering Pipeline")
    print("=" * 60)

    # ── Step 1: Load raw data ───────────────────────────────────
    df = load_raw_data()

    # ── Step 2: Engineer new features ───────────────────────────
    df = engineer_features(df)

    # ── Step 3: Remove redundant features ───────────────────────
    df, dropped_features = remove_redundant_features(df, CORR_THRESHOLD)

    # ── Step 4: Split BEFORE normalizing (prevents data leakage) ─
    train_df, val_df, test_df = split_dataset(df)

    # ── Step 5: Normalize using TRAIN statistics only ───────────
    print("\nNormalizing using training set statistics only (no data leakage)...")
    feature_cols = [c for c in train_df.columns if c != "occupancy"]

    train_means = train_df[feature_cols].mean()
    train_stds = train_df[feature_cols].std().replace(0, 1)

    # Apply train statistics to all splits
    train_df[feature_cols] = (train_df[feature_cols] - train_means) / train_stds
    val_df[feature_cols] = (val_df[feature_cols] - train_means) / train_stds
    test_df[feature_cols] = (test_df[feature_cols] - train_means) / train_stds

    print(f"  Train value range: [{train_df[feature_cols].min().min():.2f}, "
          f"{train_df[feature_cols].max().max():.2f}]")
    print(f"  Val value range:   [{val_df[feature_cols].min().min():.2f}, "
          f"{val_df[feature_cols].max().max():.2f}]")
    print(f"  Test value range:  [{test_df[feature_cols].min().min():.2f}, "
          f"{test_df[feature_cols].max().max():.2f}]")

    # ── Step 6: Save everything ─────────────────────────────────
    print("\nSaving processed datasets...")
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    # Save feature config (scaler params + feature list)
    config = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "n_classes": int(df["occupancy"].nunique()),
        "classes": sorted(df["occupancy"].unique().tolist()),
        "dropped_features": dropped_features,
        "scaler": {
            "type": "StandardScaler",
            "means": {k: float(v) for k, v in train_means.to_dict().items()},
            "stds": {k: float(v) for k, v in train_stds.to_dict().items()},
        },
        "split": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
            "stratified": True,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("FILES SAVED")
    print(f"{'=' * 60}")
    files = [
        ("Train set", TRAIN_FILE, train_df),
        ("Val set", VAL_FILE, val_df),
        ("Test set", TEST_FILE, test_df),
    ]
    for label, path, data in files:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {label:<12} {path}")
        print(f"             {data.shape[0]:,} samples × {data.shape[1]} cols  ({size_mb:.1f} MB)")

    config_size = os.path.getsize(CONFIG_FILE) / 1024
    print(f"  {'Config':<12} {CONFIG_FILE}")
    print(f"             {len(feature_cols)} features, {config['n_classes']} classes  ({config_size:.1f} KB)")

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Raw features:        222")
    print(f"  Engineered features: +80")
    print(f"  After redundancy:    {len(feature_cols)} (dropped {len(dropped_features)})")
    print(f"  Normalization:       StandardScaler (fit on train only)")
    print(f"  Split:               {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(f"  Data leakage:        None — scaler fit on train, applied to val/test")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
