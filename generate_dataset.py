import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
N_SAMPLES   = 2000
N_FEATURES  = 100
N_BENIGN    = 1000   # class 0
N_MALWARE   = 1000   # class 1

# ── Feature-group definitions (mirrors EMBER groups, scaled to 100 dims) ─────
# Each tuple: (start_idx, end_idx, group_name)
FEATURE_GROUPS = [
    ( 0,  9,  "byte_histogram"),        # 10 dims
    (10, 19,  "byte_entropy"),          # 10 dims
    (20, 29,  "string_stats"),          # 10 dims
    (30, 39,  "general_file_info"),     # 10 dims
    (40, 49,  "header_info"),           # 10 dims
    (50, 64,  "section_info"),          # 15 dims
    (65, 84,  "imports_info"),          # 20 dims
    (85, 99,  "exports_info"),          # 15 dims
]

def make_benign(n: int) -> np.ndarray:
    """
    Benign samples: low entropy, balanced byte distribution,
    few suspicious imports, moderate file size features.
    """
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # byte_histogram (0–9): roughly uniform, small variance
    X[:, 0:10]  = rng.normal(loc=0.10, scale=0.02, size=(n, 10)).clip(0, 1)

    # byte_entropy (10–19): low entropy values
    X[:, 10:20] = rng.normal(loc=0.30, scale=0.05, size=(n, 10)).clip(0, 1)

    # string_stats (20–29): moderate printable-string ratios
    X[:, 20:30] = rng.normal(loc=0.40, scale=0.08, size=(n, 10)).clip(0, 1)

    # general_file_info (30–39): typical PE sizes, small virtual-size ratios
    X[:, 30:40] = rng.normal(loc=0.35, scale=0.10, size=(n, 10)).clip(0, 1)

    # header_info (40–49): standard timestamps, compile flags
    X[:, 40:50] = rng.normal(loc=0.50, scale=0.12, size=(n, 10)).clip(0, 1)

    # section_info (50–64): normal section entropy, expected counts
    X[:, 50:65] = rng.normal(loc=0.45, scale=0.08, size=(n, 15)).clip(0, 1)

    # imports_info (65–84): common API calls present, low diversity score
    X[:, 65:85] = rng.normal(loc=0.30, scale=0.07, size=(n, 20)).clip(0, 1)

    # exports_info (85–99): few exports
    X[:, 85:100] = rng.normal(loc=0.10, scale=0.05, size=(n, 15)).clip(0, 1)

    return X


def make_malware(n: int) -> np.ndarray:
    """
    Malware samples: high entropy (packed/encrypted), skewed byte distribution,
    suspicious imports (VirtualAlloc, WriteProcessMemory …),
    many exports, abnormal section names/sizes.
    """
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # byte_histogram (0–9): skewed — high counts in upper-byte ranges
    X[:, 0:10]  = rng.normal(loc=0.70, scale=0.10, size=(n, 10)).clip(0, 1)

    # byte_entropy (10–19): high entropy (packed / encrypted payloads)
    X[:, 10:20] = rng.normal(loc=0.85, scale=0.06, size=(n, 10)).clip(0, 1)

    # string_stats (20–29): fewer printable strings, obfuscated
    X[:, 20:30] = rng.normal(loc=0.15, scale=0.06, size=(n, 10)).clip(0, 1)

    # general_file_info (30–39): unusual sizes, high virtual-to-raw ratio
    X[:, 30:40] = rng.normal(loc=0.75, scale=0.12, size=(n, 10)).clip(0, 1)

    # header_info (40–49): fake/zeroed timestamps, unusual subsystems
    X[:, 40:50] = rng.normal(loc=0.80, scale=0.10, size=(n, 10)).clip(0, 1)

    # section_info (50–64): high-entropy sections, executable + writable flags
    X[:, 50:65] = rng.normal(loc=0.82, scale=0.07, size=(n, 15)).clip(0, 1)

    # imports_info (65–84): heavy use of dangerous APIs
    X[:, 65:85] = rng.normal(loc=0.75, scale=0.09, size=(n, 20)).clip(0, 1)

    # exports_info (85–99): many exports (C2 / dropper modules)
    X[:, 85:100] = rng.normal(loc=0.70, scale=0.10, size=(n, 15)).clip(0, 1)

    # Inject sparse anomalous spikes (mimics zero-day / polymorphic behaviour)
    spike_feats = rng.integers(0, N_FEATURES, size=(n, 5))
    for i in range(n):
        X[i, spike_feats[i]] = rng.uniform(0.90, 1.00, size=5)

    return X


# ── Generate samples ──────────────────────────────────────────────────────────
X_benign  = make_benign(N_BENIGN)
X_malware = make_malware(N_MALWARE)

y_benign  = np.zeros(N_BENIGN,  dtype=np.int8)
y_malware = np.ones(N_MALWARE,  dtype=np.int8)

X = np.vstack([X_benign, X_malware])
y = np.concatenate([y_benign, y_malware])

# ── Shuffle ───────────────────────────────────────────────────────────────────
shuffle_idx = rng.permutation(N_SAMPLES)
X, y = X[shuffle_idx], y[shuffle_idx]

# ── Build DataFrame ───────────────────────────────────────────────────────────
feature_cols = [f"feature_{i:03d}" for i in range(N_FEATURES)]
df = pd.DataFrame(X, columns=feature_cols)
df.insert(0, "label", y)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = "data/malware.csv"
df.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Dataset saved  : {out_path}")
print(f"Shape          : {df.shape}")
print(f"Label counts   :\n{df['label'].value_counts().to_string()}")
print(f"\nFeature stats (first 5 cols):")
print(df[feature_cols[:5]].describe().round(4).to_string())