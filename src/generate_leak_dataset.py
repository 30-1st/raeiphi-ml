"""
RAEIPHI — Synthetic CSI Leak Detection Dataset
================================================
Generates realistic WiFi CSI data representing five scenarios:

  1. dry_empty     — empty room, no water (baseline)
  2. dry_occupied  — people present, no water
  3. leak_empty    — active leak, no people
  4. leak_occupied — active leak + people present
  5. flood         — significant water presence

Models the physics of how water affects WiFi signals differently
from human bodies:
  - Water: broadband uniform absorption, static (no temporal variance),
    progressive drift, very high dielectric constant (~80)
  - Humans: scattered absorption, high temporal variance (movement),
    non-uniform subcarrier impact

Output: ml/data/leak_dataset.csv
"""

import numpy as np
import pandas as pd
import os

# ── Config ──────────────────────────────────────────────────────
NUM_SAMPLES = 10_000
NUM_SUBCARRIERS = 52
NUM_TIME_SNAPSHOTS = 10
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "leak_dataset.csv")

np.random.seed(RANDOM_SEED)

# Class distribution
CLASS_DIST = {
    "dry_empty": 0.25,       # 2500 — baseline normal
    "dry_occupied": 0.25,    # 2500 — people, no water
    "leak_empty": 0.20,      # 2000 — leak, no people
    "leak_occupied": 0.15,   # 1500 — leak + people
    "flood": 0.15,           # 1500 — major water event
}

# Binary labels for the classifier
WATER_LABELS = {
    "dry_empty": 0,
    "dry_occupied": 0,
    "leak_empty": 1,
    "leak_occupied": 1,
    "flood": 1,
}


def generate_base_signal(n_subcarriers: int) -> np.ndarray:
    """Generate stable base CSI amplitude for a dry empty room."""
    base = np.random.uniform(20, 40, size=n_subcarriers)
    edge_rolloff = np.concatenate([
        np.linspace(0.7, 1.0, n_subcarriers // 4),
        np.ones(n_subcarriers // 2),
        np.linspace(1.0, 0.7, n_subcarriers - n_subcarriers // 4 - n_subcarriers // 2)
    ])
    return base * edge_rolloff


def generate_sample(scenario: str, base_signal: np.ndarray) -> dict:
    """
    Generate one CSI sample for a given scenario.

    Physics model:
    - DRY EMPTY: stable signal, minimal variance
    - DRY OCCUPIED: human-induced variance (from Phase 1 model)
    - LEAK EMPTY: broadband amplitude drop, very low temporal variance,
      uniform absorption across subcarriers, progressive phase drift
    - LEAK OCCUPIED: both water and human effects combined
    - FLOOD: extreme broadband absorption, very uniform, near-zero variance
    """
    n_sub = len(base_signal)
    features = {}

    # ── Determine occupancy and water parameters ────────────────
    if scenario == "dry_empty":
        n_people = 0
        water_severity = 0.0
    elif scenario == "dry_occupied":
        n_people = np.random.randint(1, 5)
        water_severity = 0.0
    elif scenario == "leak_empty":
        n_people = 0
        water_severity = np.random.uniform(0.2, 0.7)  # Partial leak
    elif scenario == "leak_occupied":
        n_people = np.random.randint(1, 3)
        water_severity = np.random.uniform(0.2, 0.7)
    elif scenario == "flood":
        n_people = 0
        water_severity = np.random.uniform(0.7, 1.0)  # Major water

    # ── Water effects on signal ─────────────────────────────────
    # Water causes broadband, UNIFORM amplitude drop
    # Unlike humans (scattered, non-uniform), water absorbs evenly
    if water_severity > 0:
        # Select affected signal paths (which subcarriers the leak sits on)
        # Water affects a contiguous band of subcarriers (spatial locality)
        leak_center = np.random.randint(5, n_sub - 5)
        leak_width = int(n_sub * (0.3 + 0.5 * water_severity))  # More water = wider effect
        leak_start = max(0, leak_center - leak_width // 2)
        leak_end = min(n_sub, leak_center + leak_width // 2)

        # Amplitude drop from water — uniform across affected subcarriers
        water_absorption = np.ones(n_sub)
        # Smooth absorption profile (not sharp edges)
        for i in range(n_sub):
            if leak_start <= i <= leak_end:
                # Uniform absorption in the affected zone
                water_absorption[i] = 1.0 - (water_severity * 0.4)
            elif abs(i - leak_start) < 3 or abs(i - leak_end) < 3:
                # Gradual transition at edges
                dist = min(abs(i - leak_start), abs(i - leak_end))
                water_absorption[i] = 1.0 - (water_severity * 0.4 * (1 - dist / 3))
    else:
        water_absorption = np.ones(n_sub)

    # ── Human effects on signal ─────────────────────────────────
    amp_variance_scale = 0.5 + 1.8 * np.sqrt(n_people) if n_people > 0 else 0.3
    absorption_factor = 1.0 - (n_people * 0.015)

    # ── Generate temporal snapshots ─────────────────────────────
    snapshots = np.zeros((NUM_TIME_SNAPSHOTS, n_sub))

    for t in range(NUM_TIME_SNAPSHOTS):
        # Base signal with water absorption (static — same every snapshot)
        water_base = base_signal * water_absorption * absorption_factor

        # Human-induced noise (varies per snapshot — movement)
        human_noise = np.random.normal(0, amp_variance_scale, size=n_sub)
        if n_people > 0:
            n_affected = min(n_sub, int(n_sub * 0.3 * np.sqrt(n_people)))
            affected = np.random.choice(n_sub, n_affected, replace=False)
            human_noise[affected] *= np.random.uniform(1.5, 3.0, size=n_affected)

        # Water-induced noise — VERY low variance (static presence)
        water_noise = np.random.normal(0, 0.1 * water_severity, size=n_sub)

        # Environmental noise (always present, small)
        env_noise = np.random.normal(0, 0.3, size=n_sub)

        snapshots[t] = water_base + human_noise + water_noise + env_noise

    # ── Amplitude features ──────────────────────────────────────
    amp_mean = np.mean(snapshots, axis=0)
    amp_std = np.std(snapshots, axis=0)
    amp_max = np.max(snapshots, axis=0)
    amp_min = np.min(snapshots, axis=0)
    amp_range = amp_max - amp_min

    for i in range(n_sub):
        features[f"amp_mean_sc{i}"] = amp_mean[i]
        features[f"amp_std_sc{i}"] = amp_std[i]
        features[f"amp_range_sc{i}"] = amp_range[i]

    # Global amplitude stats
    features["amp_mean_global"] = np.mean(amp_mean)
    features["amp_std_global"] = np.mean(amp_std)
    features["amp_range_global"] = np.mean(amp_range)
    features["amp_std_max"] = np.max(amp_std)
    features["amp_std_min"] = np.min(amp_std)
    features["amp_variance_spread"] = np.max(amp_std) - np.min(amp_std)

    # ── Phase data ──────────────────────────────────────────────
    base_phase = np.random.uniform(-np.pi, np.pi, size=n_sub)
    human_phase_instability = 0.05 + 0.25 * np.sqrt(n_people) if n_people > 0 else 0.03
    # Water causes SUSTAINED phase shift, not instability
    water_phase_shift = water_severity * 0.8  # Progressive static shift

    phase_snapshots = np.zeros((NUM_TIME_SNAPSHOTS, n_sub))
    for t in range(NUM_TIME_SNAPSHOTS):
        human_phase_noise = np.random.normal(0, human_phase_instability, size=n_sub)
        # Water phase shift is constant (not noisy like humans)
        water_phase = np.zeros(n_sub)
        if water_severity > 0:
            water_phase[leak_start:leak_end] = water_phase_shift
        phase_snapshots[t] = base_phase + human_phase_noise + water_phase

    phase_std = np.std(phase_snapshots, axis=0)
    for i in range(n_sub):
        features[f"phase_std_sc{i}"] = phase_std[i]

    features["phase_std_global"] = np.mean(phase_std)
    features["phase_std_max"] = np.max(phase_std)
    features["phase_std_spread"] = np.max(phase_std) - np.min(phase_std)

    # ── Temporal dynamics ───────────────────────────────────────
    temporal_diffs = np.diff(snapshots, axis=0)
    features["temporal_diff_mean"] = np.mean(np.abs(temporal_diffs))
    features["temporal_diff_max"] = np.max(np.abs(temporal_diffs))
    features["temporal_diff_std"] = np.std(temporal_diffs)

    # ── Cross-subcarrier correlation ────────────────────────────
    if NUM_TIME_SNAPSHOTS > 1:
        corr_matrix = np.corrcoef(snapshots.T)
        upper_tri = corr_matrix[np.triu_indices(n_sub, k=1)]
        features["subcarrier_corr_mean"] = np.mean(upper_tri)
        features["subcarrier_corr_std"] = np.std(upper_tri)

    # ── WATER-SPECIFIC FEATURES ─────────────────────────────────
    # These distinguish water from human presence

    # 1. Absorption uniformity — how uniform is the amplitude drop across subcarriers
    # Water: very uniform. Humans: scattered.
    amp_drop = base_signal - amp_mean  # How much amplitude dropped from baseline
    if np.std(amp_drop) > 0:
        features["absorption_uniformity"] = 1.0 - (np.std(amp_drop) / (np.mean(np.abs(amp_drop)) + 1e-6))
    else:
        features["absorption_uniformity"] = 1.0

    # 2. Temporal stability ratio — low temporal variance relative to amplitude drop
    # Water: high stability (static). Humans: low stability (moving).
    mean_amp_drop = np.mean(np.abs(amp_drop))
    mean_temporal_var = np.mean(amp_std)
    features["temporal_stability_ratio"] = mean_amp_drop / (mean_temporal_var + 1e-6)

    # 3. Broadband absorption score — fraction of subcarriers showing amplitude drop
    # Water: affects many subcarriers broadly. Humans: affect fewer, scattered.
    drop_threshold = 0.5  # dB
    features["broadband_absorption_score"] = np.sum(amp_drop > drop_threshold) / n_sub

    # 4. Amplitude drop magnitude — total signal loss
    features["amplitude_drop_magnitude"] = float(np.mean(amp_drop))

    # 5. Phase drift consistency — is the phase shift static or varying?
    # Water: consistent drift. Humans: random variation.
    phase_drift = np.mean(phase_snapshots, axis=0) - base_phase
    features["phase_drift_magnitude"] = float(np.mean(np.abs(phase_drift)))
    features["phase_drift_consistency"] = 1.0 - (np.std(np.abs(phase_drift)) / (np.mean(np.abs(phase_drift)) + 1e-6))

    # 6. Signal-to-noise ratio change
    snr_values = amp_mean / (amp_std + 1e-6)
    features["snr_global_mean"] = float(np.mean(snr_values))
    features["snr_global_std"] = float(np.std(snr_values))

    # 7. Affected subcarrier contiguity — water affects contiguous bands, humans don't
    significant_drops = amp_drop > 1.0
    if np.sum(significant_drops) > 0:
        # Count longest run of consecutive affected subcarriers
        runs = np.diff(np.where(np.concatenate(([significant_drops[0]],
                       significant_drops[:-1] != significant_drops[1:],
                       [True])))[0])[::2]
        features["max_contiguous_affected"] = int(np.max(runs)) if len(runs) > 0 else 0
        features["contiguity_ratio"] = features["max_contiguous_affected"] / max(np.sum(significant_drops), 1)
    else:
        features["max_contiguous_affected"] = 0
        features["contiguity_ratio"] = 0.0

    # ── Labels ──────────────────────────────────────────────────
    features["scenario"] = scenario
    features["water_present"] = WATER_LABELS[scenario]
    features["water_severity"] = water_severity
    features["n_people"] = n_people

    return features


def main():
    print("=" * 60)
    print("RAEIPHI — Leak Detection CSI Dataset Generator")
    print("=" * 60)

    base_signal = generate_base_signal(NUM_SUBCARRIERS)
    print(f"Base signal: {NUM_SUBCARRIERS} subcarriers")
    print(f"Amplitude range: {base_signal.min():.1f} – {base_signal.max():.1f} dB")

    # Calculate sample counts per class
    class_counts = {}
    remaining = NUM_SAMPLES
    scenarios = list(CLASS_DIST.keys())
    for i, scenario in enumerate(scenarios):
        if i == len(scenarios) - 1:
            class_counts[scenario] = remaining
        else:
            count = int(NUM_SAMPLES * CLASS_DIST[scenario])
            class_counts[scenario] = count
            remaining -= count

    print(f"\nClass distribution ({NUM_SAMPLES} total):")
    for scenario, count in class_counts.items():
        water_label = "WET" if WATER_LABELS[scenario] else "DRY"
        bar = "█" * int(count / NUM_SAMPLES * 80)
        print(f"  {scenario:<16} [{water_label}]: {count:>5}  {bar}")

    # Generate samples
    print(f"\nGenerating samples...")
    all_samples = []
    sample_count = 0

    for scenario, count in class_counts.items():
        for _ in range(count):
            sample = generate_sample(scenario, base_signal)
            all_samples.append(sample)
            sample_count += 1
            if sample_count % 2000 == 0:
                print(f"  {sample_count}/{NUM_SAMPLES}...")

    # Build DataFrame and shuffle
    df = pd.DataFrame(all_samples)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    n_features = len([c for c in df.columns if c not in
                      ["scenario", "water_present", "water_severity", "n_people"]])

    print(f"\n{'=' * 60}")
    print(f"Dataset saved: {OUTPUT_FILE}")
    print(f"  Samples:        {len(df):,}")
    print(f"  Features:       {n_features}")
    print(f"  Labels:         'water_present' (0=dry, 1=wet)")
    print(f"  Scenarios:      {df['scenario'].nunique()}")
    print(f"  File size:      {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")
    print(f"\n  Binary label distribution:")
    for label, count in df["water_present"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        name = "DRY" if label == 0 else "WET"
        print(f"    {name} ({label}): {count:,} ({pct:.1f}%)")
    print(f"\n  Scenario distribution:")
    for scenario, count in df["scenario"].value_counts().items():
        print(f"    {scenario:<16}: {count:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
