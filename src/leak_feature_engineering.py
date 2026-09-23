"""
RAEIPHI — Leak Detection Feature Engineering
==============================================
Loads the leak dataset, engineers water-specific features,
removes redundant features, normalizes, and splits into
train/val/test sets.

Input:  ml/data/leak_dataset.csv
Output: ml/data/leak_train.csv, leak_val.csv, leak_test.csv
        ml/data/leak_feature_config.json
"""

import pandas as pd
import numpy as np
import json
import os

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "leak_dataset.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "leak_train.csv")
VAL_FILE = os.path.join(DATA_DIR, "leak_val.csv")
TEST_FILE = os.path.join(DATA_DIR, "leak_test.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "leak_feature_config.json")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CORR_THRESHOLD = 0.95

np.random.seed(RANDOM_SEED)

LABEL_COL = "water_present"
META_COLS = ["scenario", "water_present", "water_severity", "n_people"]


def load_data():
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")
    return df


def engineer_features(df):
    """Engineer additional water-detection features."""
    print("\nEngineering water-specific features...")
    n_sub = 52

    # SNR per subcarrier
    for i in range(n_sub):
        df[f"snr_sc{i}"] = df[f"amp_mean_sc{i}"] / (df[f"amp_std_sc{i}"] + 1e-6)

    snr_cols = [f"snr_sc{i}" for i in range(n_sub)]
    df["snr_min"] = df[snr_cols].min(axis=1)
    df["snr_max"] = df[snr_cols].max(axis=1)
    print("  + 52 per-subcarrier SNR + 2 global SNR stats")

    # Amplitude distribution shape
    amp_std_cols = [f"amp_std_sc{i}" for i in range(n_sub)]
    amp_mean_cols = [f"amp_mean_sc{i}" for i in range(n_sub)]

    df["amp_std_skew"] = df[amp_std_cols].skew(axis=1)
    df["amp_std_kurtosis"] = df[amp_std_cols].kurtosis(axis=1)
    df["amp_mean_skew"] = df[amp_mean_cols].skew(axis=1)
    df["amp_mean_kurtosis"] = df[amp_mean_cols].kurtosis(axis=1)
    print("  + 4 distribution shape features")

    # Band analysis (water affects contiguous bands)
    for bname, idxs in [("lower", range(0, 17)), ("mid", range(17, 35)), ("upper", range(35, 52))]:
        band_std = [f"amp_std_sc{i}" for i in idxs]
        band_mean = [f"amp_mean_sc{i}" for i in idxs]
        df[f"amp_std_{bname}_mean"] = df[band_std].mean(axis=1)
        df[f"amp_mean_{bname}_mean"] = df[band_mean].mean(axis=1)

    # Inter-band amplitude difference (water causes uniform drop, so bands should be similar)
    df["band_amplitude_variance"] = df[["amp_mean_lower_mean", "amp_mean_mid_mean", "amp_mean_upper_mean"]].std(axis=1)
    print("  + 7 band features")

    # Percentiles of amplitude std (water = low variance, humans = high)
    df["amp_std_p10"] = df[amp_std_cols].quantile(0.10, axis=1)
    df["amp_std_p50"] = df[amp_std_cols].quantile(0.50, axis=1)
    df["amp_std_p90"] = df[amp_std_cols].quantile(0.90, axis=1)
    df["amp_std_iqr"] = df["amp_std_p90"] - df["amp_std_p10"]
    print("  + 4 amplitude std percentiles")

    # Ratio features
    df["phase_to_temporal_ratio"] = df["phase_std_global"] / (df["temporal_diff_mean"] + 1e-6)
    df["drop_to_variance_ratio"] = df["amplitude_drop_magnitude"] / (df["amp_std_global"] + 1e-6)
    print("  + 2 ratio features")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"\n  Total features before selection: {len(feature_cols)}")
    return df


def remove_redundant(df, threshold):
    """Remove highly correlated features."""
    print(f"\nRemoving features with correlation > {threshold}...")
    feature_cols = [c for c in df.columns if c not in META_COLS]

    corr = df[feature_cols].corr().abs()
    target_corr = df[feature_cols].corrwith(df[LABEL_COL]).abs()

    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr.iloc[i, j] > threshold:
                col_i, col_j = feature_cols[i], feature_cols[j]
                if target_corr.get(col_i, 0) < target_corr.get(col_j, 0):
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)

    df = df.drop(columns=list(to_drop))
    remaining = len([c for c in df.columns if c not in META_COLS])
    print(f"  Dropped {len(to_drop)}, remaining: {remaining}")
    return df, list(to_drop)


def split_and_normalize(df):
    """Stratified split and StandardScaler normalization."""
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Stratified split
    print(f"\nSplitting ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")
    train_dfs, val_dfs, test_dfs = [], [], []

    for label in sorted(df[LABEL_COL].unique()):
        class_df = df[df[LABEL_COL] == label].sample(frac=1, random_state=RANDOM_SEED)
        n = len(class_df)
        t_end = int(n * TRAIN_RATIO)
        v_end = t_end + int(n * VAL_RATIO)
        train_dfs.append(class_df.iloc[:t_end])
        val_dfs.append(class_df.iloc[t_end:v_end])
        test_dfs.append(class_df.iloc[v_end:])

    train = pd.concat(train_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val = pd.concat(val_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test = pd.concat(test_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # Normalize using train stats only
    print("Normalizing (fit on train only)...")
    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1)

    train[feature_cols] = (train[feature_cols] - means) / stds
    val[feature_cols] = (val[feature_cols] - means) / stds
    test[feature_cols] = (test[feature_cols] - means) / stds

    scaler = {
        "means": {k: float(v) for k, v in means.to_dict().items()},
        "stds": {k: float(v) for k, v in stds.to_dict().items()},
    }

    return train, val, test, feature_cols, scaler


def main():
    print("=" * 60)
    print("RAEIPHI — Leak Detection Feature Engineering")
    print("=" * 60)

    df = load_data()
    df = engineer_features(df)
    df, dropped = remove_redundant(df, CORR_THRESHOLD)
    train, val, test, feature_cols, scaler = split_and_normalize(df)

    # Save splits
    train.to_csv(TRAIN_FILE, index=False)
    val.to_csv(VAL_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    # Save config
    config = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "label_column": LABEL_COL,
        "n_classes": 2,
        "classes": [0, 1],
        "class_names": ["dry", "wet"],
        "dropped_features": dropped,
        "scaler": {"type": "StandardScaler", **scaler},
        "split": {
            "train": len(train), "val": len(val), "test": len(test),
            "seed": RANDOM_SEED,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Features:   {len(feature_cols)}")
    print(f"  Train:      {TRAIN_FILE}")
    print(f"  Val:        {VAL_FILE}")
    print(f"  Test:       {TEST_FILE}")
    print(f"  Config:     {CONFIG_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
