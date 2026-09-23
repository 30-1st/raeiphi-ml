"""
RAEIPHI — Synthetic CSI Dataset Generator
==========================================
Generates realistic WiFi Channel State Information (CSI) data
labeled by room occupancy count (0–10 people).

CSI captures amplitude and phase across WiFi subcarriers.
Human presence disturbs the signal through reflection, absorption,
and scattering — more people = more disturbance = higher variance
in amplitude readings and more phase instability.

Output: ml/data/csi_dataset.csv (10,000 samples)
"""

import numpy as np
import pandas as pd
import os

# ── Config ──────────────────────────────────────────────────────
NUM_SAMPLES = 10_000
NUM_SUBCARRIERS = 52          # Standard 20MHz 802.11n
NUM_TIME_SNAPSHOTS = 10       # Temporal window per sample
MAX_OCCUPANCY = 10
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "csi_dataset.csv")

np.random.seed(RANDOM_SEED)


def get_class_distribution(n_samples: int) -> dict:
    """
    Realistic occupancy distribution for short-term rentals.
    Most readings are 0–3 people. 4–6 is uncommon. 7–10 is rare
    (party scenario). Skewed distribution reflects real-world usage.
    """
    weights = {
        0: 0.18,   # empty — between bookings, daytime vacancy
        1: 0.20,   # solo traveler
        2: 0.22,   # couple — most common booking type
        3: 0.14,   # small family / friends
        4: 0.09,   # family with kids
        5: 0.06,   # small group
        6: 0.04,   # larger group
        7: 0.03,   # exceeding typical booking
        8: 0.02,   # party territory
        9: 0.01,   # party
        10: 0.01,  # party
    }
    counts = {}
    remaining = n_samples
    classes = sorted(weights.keys())
    for i, cls in enumerate(classes):
        if i == len(classes) - 1:
            counts[cls] = remaining
        else:
            counts[cls] = int(n_samples * weights[cls])
            remaining -= counts[cls]
    return counts


def generate_base_signal(n_subcarriers: int) -> np.ndarray:
    """
    Generate a stable base CSI amplitude profile for an empty room.
    Each subcarrier has a characteristic base amplitude determined
    by the room geometry, wall materials, and antenna positioning.
    """
    # Base amplitude per subcarrier (dB scale, typical range 15–45 dB)
    base = np.random.uniform(20, 40, size=n_subcarriers)
    # Subcarriers near edges of the band tend to be weaker
    edge_rolloff = np.concatenate([
        np.linspace(0.7, 1.0, n_subcarriers // 4),
        np.ones(n_subcarriers // 2),
        np.linspace(1.0, 0.7, n_subcarriers - n_subcarriers // 4 - n_subcarriers // 2)
    ])
    return base * edge_rolloff


def generate_sample(occupancy: int, base_signal: np.ndarray) -> dict:
    """
    Generate one CSI sample for a given occupancy count.

    Physics model (simplified):
    - Each person adds multipath reflections → amplitude fluctuation
    - More people → higher temporal variance (movement)
    - More people → more phase instability
    - Body absorption reduces mean amplitude slightly
    - People cluster effects: variance doesn't scale perfectly linearly
    """
    n_sub = len(base_signal)
    features = {}

    # ── Amplitude across time snapshots ─────────────────────────
    # Each person adds signal disturbance
    # Amplitude variance scales with sqrt(occupancy) — diminishing per-person effect
    amp_variance_scale = 0.5 + 1.8 * np.sqrt(occupancy)

    # Body absorption: each person reduces mean amplitude slightly
    absorption_factor = 1.0 - (occupancy * 0.015)  # ~1.5% drop per person

    # Generate temporal snapshots of amplitude readings
    snapshots = np.zeros((NUM_TIME_SNAPSHOTS, n_sub))
    for t in range(NUM_TIME_SNAPSHOTS):
        noise = np.random.normal(0, amp_variance_scale, size=n_sub)
        # Movement-induced fading: random subcarriers get hit harder
        if occupancy > 0:
            n_affected = min(n_sub, int(n_sub * 0.3 * np.sqrt(occupancy)))
            affected_subs = np.random.choice(n_sub, n_affected, replace=False)
            noise[affected_subs] *= np.random.uniform(1.5, 3.0, size=n_affected)
        snapshots[t] = base_signal * absorption_factor + noise

    # ── Aggregate amplitude features ────────────────────────────
    amp_mean = np.mean(snapshots, axis=0)      # Mean amplitude per subcarrier
    amp_std = np.std(snapshots, axis=0)         # Temporal variance per subcarrier
    amp_max = np.max(snapshots, axis=0)
    amp_min = np.min(snapshots, axis=0)
    amp_range = amp_max - amp_min               # Peak-to-peak variation

    # Store per-subcarrier amplitude features
    for i in range(n_sub):
        features[f"amp_mean_sc{i}"] = amp_mean[i]
        features[f"amp_std_sc{i}"] = amp_std[i]
        features[f"amp_range_sc{i}"] = amp_range[i]

    # ── Global amplitude statistics ─────────────────────────────
    features["amp_mean_global"] = np.mean(amp_mean)
    features["amp_std_global"] = np.mean(amp_std)
    features["amp_range_global"] = np.mean(amp_range)
    features["amp_std_max"] = np.max(amp_std)
    features["amp_std_min"] = np.min(amp_std)
    features["amp_variance_spread"] = np.max(amp_std) - np.min(amp_std)

    # ── Phase data ──────────────────────────────────────────────
    # Phase is measured in radians (-π to π)
    # Empty room: phase is stable. People cause phase shifts.
    base_phase = np.random.uniform(-np.pi, np.pi, size=n_sub)
    phase_instability = 0.05 + 0.25 * np.sqrt(occupancy)

    phase_snapshots = np.zeros((NUM_TIME_SNAPSHOTS, n_sub))
    for t in range(NUM_TIME_SNAPSHOTS):
        phase_noise = np.random.normal(0, phase_instability, size=n_sub)
        phase_snapshots[t] = base_phase + phase_noise

    phase_std = np.std(phase_snapshots, axis=0)

    # Store per-subcarrier phase variance
    for i in range(n_sub):
        features[f"phase_std_sc{i}"] = phase_std[i]

    # Global phase statistics
    features["phase_std_global"] = np.mean(phase_std)
    features["phase_std_max"] = np.max(phase_std)
    features["phase_std_spread"] = np.max(phase_std) - np.min(phase_std)

    # ── Cross-subcarrier correlation ────────────────────────────
    # More people → less correlated subcarrier behavior
    # (each person's reflection pattern is different)
    if NUM_TIME_SNAPSHOTS > 1:
        corr_matrix = np.corrcoef(snapshots.T)
        upper_tri = corr_matrix[np.triu_indices(n_sub, k=1)]
        features["subcarrier_corr_mean"] = np.mean(upper_tri)
        features["subcarrier_corr_std"] = np.std(upper_tri)
    else:
        features["subcarrier_corr_mean"] = 1.0
        features["subcarrier_corr_std"] = 0.0

    # ── Temporal dynamics ───────────────────────────────────────
    # Consecutive snapshot differences — captures movement rate
    temporal_diffs = np.diff(snapshots, axis=0)
    features["temporal_diff_mean"] = np.mean(np.abs(temporal_diffs))
    features["temporal_diff_max"] = np.max(np.abs(temporal_diffs))
    features["temporal_diff_std"] = np.std(temporal_diffs)

    # ── Label ───────────────────────────────────────────────────
    features["occupancy"] = occupancy

    return features


def main():
    print("=" * 60)
    print("RAEIPHI — CSI Dataset Generator")
    print("=" * 60)

    # Generate base room signal profile
    base_signal = generate_base_signal(NUM_SUBCARRIERS)
    print(f"Base signal profile: {NUM_SUBCARRIERS} subcarriers")
    print(f"Amplitude range: {base_signal.min():.1f} – {base_signal.max():.1f} dB")

    # Get class distribution
    class_dist = get_class_distribution(NUM_SAMPLES)
    print(f"\nClass distribution ({NUM_SAMPLES} total samples):")
    for occ, count in sorted(class_dist.items()):
        bar = "█" * int(count / NUM_SAMPLES * 100)
        print(f"  Occupancy {occ:>2d}: {count:>5d} samples  {bar}")

    # Generate all samples
    print(f"\nGenerating samples...")
    all_samples = []
    sample_count = 0

    for occupancy, count in sorted(class_dist.items()):
        for _ in range(count):
            sample = generate_sample(occupancy, base_signal)
            all_samples.append(sample)
            sample_count += 1
            if sample_count % 2000 == 0:
                print(f"  {sample_count}/{NUM_SAMPLES} samples generated...")

    # Build DataFrame
    df = pd.DataFrame(all_samples)

    # Shuffle rows (samples were generated in class order)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    n_features = len(df.columns) - 1  # minus the label column
    print(f"\n{'=' * 60}")
    print(f"Dataset saved: {OUTPUT_FILE}")
    print(f"  Samples:    {len(df):,}")
    print(f"  Features:   {n_features}")
    print(f"  Label:      'occupancy' (0–{MAX_OCCUPANCY})")
    print(f"  File size:  {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")

    # Feature breakdown
    amp_mean_cols = [c for c in df.columns if c.startswith("amp_mean_sc")]
    amp_std_cols = [c for c in df.columns if c.startswith("amp_std_sc")]
    amp_range_cols = [c for c in df.columns if c.startswith("amp_range_sc")]
    phase_std_cols = [c for c in df.columns if c.startswith("phase_std_sc")]
    global_cols = [c for c in df.columns if not c.startswith(("amp_mean_sc", "amp_std_sc", "amp_range_sc", "phase_std_sc")) and c != "occupancy"]

    print(f"\nFeature breakdown:")
    print(f"  Per-subcarrier amplitude mean:  {len(amp_mean_cols)} features")
    print(f"  Per-subcarrier amplitude std:   {len(amp_std_cols)} features")
    print(f"  Per-subcarrier amplitude range: {len(amp_range_cols)} features")
    print(f"  Per-subcarrier phase std:       {len(phase_std_cols)} features")
    print(f"  Global / aggregate features:    {len(global_cols)} features")
    print(f"  Total:                          {n_features} features")


if __name__ == "__main__":
    main()
