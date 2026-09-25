# ============================================================
# SAAD Evidence Engine v2
# ============================================================
#
# Purpose:
#   Convert frozen YOLO + VAE + Flow image-level evidence into
#   an interpretable review-priority / evidence-ranking system.
#
# IMPORTANT:
#   - No training
#   - No test-label fitting
#   - No arbitrary NORMAL/ANOMALY probability
#   - Normalization is fitted ONLY from validation normals
#   - Test labels are used ONLY for post-hoc evaluation
#
# Inputs:
#   E:\SIH\SIH_Results\validation_weighted_fusion\
#       validation_raw_scores.csv
#
#   E:\SIH\SIH_Results\validation_weighted_fusion\
#       test_raw_scores.csv
#
# Outputs:
#   E:\SIH\SIH_Results\saad_evidence_engine_v2\
#
# ============================================================

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SIH\SIH_Results")

INPUT_DIR = BASE_DIR / "validation_weighted_fusion"

VAL_CSV = INPUT_DIR / "validation_raw_scores.csv"
TEST_CSV = INPUT_DIR / "test_raw_scores.csv"

OUT_DIR = BASE_DIR / "saad_evidence_engine_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Evidence weights
# ------------------------------------------------------------
#
# YOLO is intentionally dominant because the untouched test
# showed that YOLO remains the strongest individual signal.
#
# VAE + Flow are corroborating normality evidence.
#
# These are NOT learned probabilities.
# ------------------------------------------------------------

W_YOLO = 0.50
W_VAE = 0.20
W_FLOW = 0.30


# ------------------------------------------------------------
# Normalization percentiles
# ------------------------------------------------------------

LOW_PCT = 5.0
HIGH_PCT = 95.0


# ------------------------------------------------------------
# Evidence bands
#
# These are descriptive evidence levels, not calibrated
# probabilities.
# ------------------------------------------------------------

WEAK = 0.33
MODERATE = 0.66


# ------------------------------------------------------------
# Search/review queue parameters
# ------------------------------------------------------------

# Small bonus for corroboration between independent signals.
CORROBORATION_BONUS = 0.10

# Penalty when YOLO is extremely weak and anomaly models are
# also weak.
LOW_EVIDENCE_PENALTY = 0.10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def finite_float(x, default=np.nan):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def minmax_percentile_normalize(x, low, high):
    """
    Normalize score so that:
        low  -> 0
        high -> 1

    Values outside range are clipped.

    This is NOT a probability.
    """
    x = np.asarray(x, dtype=np.float64)

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Normalization bounds are non-finite.")

    if high <= low:
        raise ValueError(
            f"Invalid normalization range: low={low}, high={high}"
        )

    z = (x - low) / (high - low)
    z = np.clip(z, 0.0, 1.0)

    return z


def evidence_band(score):
    if not np.isfinite(score):
        return "UNKNOWN"

    if score < WEAK:
        return "LOW"

    if score < MODERATE:
        return "MODERATE"

    return "STRONG"


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.std(values))


def binary_metrics(y_true, score):
    """
    Post-hoc metrics only.
    These are NEVER used to fit the engine.
    """

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    mask = np.isfinite(score)

    y = y_true[mask]
    s = score[mask]

    result = {
        "n": len(y),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "best_f1": np.nan,
        "best_threshold": np.nan,
        "best_precision": np.nan,
        "best_recall": np.nan,
    }

    if len(np.unique(y)) < 2:
        return result

    result["roc_auc"] = float(roc_auc_score(y, s))
    result["pr_auc"] = float(average_precision_score(y, s))

    precision, recall, thresholds = precision_recall_curve(y, s)

    if len(thresholds) == 0:
        return result

    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    idx = int(np.nanargmax(f1))

    result["best_f1"] = float(f1[idx])
    result["best_threshold"] = float(thresholds[idx])
    result["best_precision"] = float(precision[idx])
    result["best_recall"] = float(recall[idx])

    return result


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAAD EVIDENCE ENGINE v2")
print("=" * 70)

print()
print("Validation:", VAL_CSV)
print("Test      :", TEST_CSV)
print("Output    :", OUT_DIR)
print()


if not VAL_CSV.exists():
    raise FileNotFoundError(f"Validation CSV not found:\n{VAL_CSV}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found:\n{TEST_CSV}")


val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)

print(f"Validation rows : {len(val)}")
print(f"Test rows       : {len(test)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "output_image",
    "path",
    "dataset",
    "group",
    "label",
    "yolo_raw",
    "vae_raw",
    "flow_raw",
]

missing_val = [c for c in required if c not in val.columns]
missing_test = [c for c in required if c not in test.columns]

if missing_val:
    raise ValueError(
        "Validation CSV missing columns:\n"
        + "\n".join(missing_val)
    )

if missing_test:
    raise ValueError(
        "Test CSV missing columns:\n"
        + "\n".join(missing_test)
    )


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

for df in [val, test]:

    for col in [
        "label",
        "yolo_raw",
        "vae_raw",
        "flow_raw",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["label"] = df["label"].fillna(0).astype(int)


# ============================================================
# FIT NORMALIZATION
# ============================================================
#
# ONLY VALIDATION NORMALS.
#
# This is critical.
#
# No test labels enter these parameters.
# ============================================================

val_norm = val[val["label"] == 0].copy()

print()
print("=" * 70)
print("VALIDATION NORMAL REFERENCE")
print("=" * 70)

print(f"Validation normal images: {len(val_norm)}")


normalization = {}

for col in [
    "yolo_raw",
    "vae_raw",
    "flow_raw",
]:

    values = val_norm[col].dropna().values.astype(float)

    low = float(np.percentile(values, LOW_PCT))
    high = float(np.percentile(values, HIGH_PCT))

    normalization[col] = {
        "p_low": low,
        "p_high": high,
        "n": len(values),
    }

    print(
        f"{col:10s} "
        f"P{LOW_PCT:g}={low:.8f} "
        f"P{HIGH_PCT:g}={high:.8f}"
    )


# Save normalization parameters

norm_rows = []

for col, params in normalization.items():

    norm_rows.append({
        "signal": col,
        "reference": "validation_normals_only",
        "low_percentile": LOW_PCT,
        "high_percentile": HIGH_PCT,
        "low_value": params["p_low"],
        "high_value": params["p_high"],
        "n_reference": params["n"],
    })

pd.DataFrame(norm_rows).to_csv(
    OUT_DIR / "normalization_parameters.csv",
    index=False,
)


# ============================================================
# NORMALIZE BOTH VALIDATION AND TEST
# ============================================================

def apply_normalization(df):

    out = df.copy()

    out["yolo_evidence"] = minmax_percentile_normalize(
        out["yolo_raw"].values,
        normalization["yolo_raw"]["p_low"],
        normalization["yolo_raw"]["p_high"],
    )

    out["vae_evidence"] = minmax_percentile_normalize(
        out["vae_raw"].values,
        normalization["vae_raw"]["p_low"],
        normalization["vae_raw"]["p_high"],
    )

    out["flow_evidence"] = minmax_percentile_normalize(
        out["flow_raw"].values,
        normalization["flow_raw"]["p_low"],
        normalization["flow_raw"]["p_high"],
    )

    return out


val_e = apply_normalization(val)
test_e = apply_normalization(test)


# ============================================================
# EVIDENCE FUSION
# ============================================================

def build_evidence(df):

    out = df.copy()

    y = out["yolo_evidence"].values
    v = out["vae_evidence"].values
    f = out["flow_evidence"].values

    # --------------------------------------------------------
    # Primary composite score
    # --------------------------------------------------------

    base = (
        W_YOLO * y
        + W_VAE * v
        + W_FLOW * f
    )

    # --------------------------------------------------------
    # Corroboration
    #
    # Independent normality evidence agrees when both VAE
    # and Flow are high.
    # --------------------------------------------------------

    normality_min = np.minimum(v, f)

    corroboration = (
        (normality_min >= MODERATE)
        .astype(float)
    )

    # --------------------------------------------------------
    # YOLO + normality corroboration
    # --------------------------------------------------------

    detector_strong = y >= MODERATE
    normality_strong = normality_min >= MODERATE

    both_strong = (
        detector_strong
        & normality_strong
    )

    priority = (
        base
        + CORROBORATION_BONUS * both_strong.astype(float)
    )

    # --------------------------------------------------------
    # If all signals are weak, slightly suppress priority.
    # --------------------------------------------------------

    all_weak = (
        (y < WEAK)
        & (v < WEAK)
        & (f < WEAK)
    )

    priority = (
        priority
        - LOW_EVIDENCE_PENALTY * all_weak.astype(float)
    )

    priority = np.clip(priority, 0.0, 1.0)

    out["base_evidence_score"] = base
    out["corroboration"] = corroboration
    out["review_priority"] = priority

    # --------------------------------------------------------
    # Individual evidence bands
    # --------------------------------------------------------

    out["yolo_band"] = [
        evidence_band(x) for x in y
    ]

    out["vae_band"] = [
        evidence_band(x) for x in v
    ]

    out["flow_band"] = [
        evidence_band(x) for x in f
    ]

    # --------------------------------------------------------
    # Evidence profile
    # --------------------------------------------------------

    profiles = []

    for yy, vv, ff, corr in zip(
        y,
        v,
        f,
        corroboration,
    ):

        y_strong = yy >= MODERATE
        v_strong = vv >= MODERATE
        f_strong = ff >= MODERATE

        y_weak = yy < WEAK
        v_weak = vv < WEAK
        f_weak = ff < WEAK

        if y_strong and v_strong and f_strong:
            profile = "STRONG_MULTI_SIGNAL"

        elif y_strong and (v_strong or f_strong):
            profile = "DETECTOR_PLUS_NORMALITY"

        elif y_strong:
            profile = "DETECTOR_DOMINANT"

        elif (v_strong and f_strong):
            profile = "NORMALITY_CORROBORATED"

        elif v_strong or f_strong:
            profile = "SINGLE_NORMALITY_SIGNAL"

        elif y_weak and v_weak and f_weak:
            profile = "LOW_EVIDENCE"

        else:
            profile = "AMBIGUOUS"

        profiles.append(profile)

    out["evidence_profile"] = profiles

    # --------------------------------------------------------
    # Recommended review action
    #
    # This is intentionally a PRIORITIZATION policy.
    # It is NOT an anomaly classifier.
    # --------------------------------------------------------

    actions = []

    for priority, profile, yy in zip(
        priority,
        profiles,
        y,
    ):

        if (
            priority >= 0.70
            and profile in [
                "STRONG_MULTI_SIGNAL",
                "DETECTOR_PLUS_NORMALITY",
            ]
        ):
            action = "HIGH_PRIORITY_REVIEW"

        elif priority >= 0.50:
            action = "REVIEW"

        elif priority >= 0.30:
            action = "LOW_PRIORITY_REVIEW"

        else:
            action = "DEFER"

        actions.append(action)

    out["recommended_action"] = actions

    return out


val_e = build_evidence(val_e)
test_e = build_evidence(test_e)


# ============================================================
# RANKING
# ============================================================

val_e = val_e.sort_values(
    "review_priority",
    ascending=False,
).reset_index(drop=True)

test_e = test_e.sort_values(
    "review_priority",
    ascending=False,
).reset_index(drop=True)

val_e["rank"] = np.arange(1, len(val_e) + 1)
test_e["rank"] = np.arange(1, len(test_e) + 1)


# ============================================================
# SAVE FULL OUTPUTS
# ============================================================

val_e.to_csv(
    OUT_DIR / "validation_evidence_queue.csv",
    index=False,
)

test_e.to_csv(
    OUT_DIR / "test_evidence_queue.csv",
    index=False,
)


# ============================================================
# EVIDENCE PROFILE SUMMARY
# ============================================================

def profile_summary(df):

    rows = []

    for profile, g in df.groupby(
        "evidence_profile",
        dropna=False,
    ):

        rows.append({
            "evidence_profile": profile,
            "n": len(g),
            "mean_priority": safe_mean(
                g["review_priority"]
            ),
            "median_priority": safe_median(
                g["review_priority"]
            ),
            "mean_yolo_evidence": safe_mean(
                g["yolo_evidence"]
            ),
            "mean_vae_evidence": safe_mean(
                g["vae_evidence"]
            ),
            "mean_flow_evidence": safe_mean(
                g["flow_evidence"]
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "mean_priority",
        ascending=False,
    )


val_profile = profile_summary(val_e)
test_profile = profile_summary(test_e)

val_profile.to_csv(
    OUT_DIR / "validation_profile_summary.csv",
    index=False,
)

test_profile.to_csv(
    OUT_DIR / "test_profile_summary.csv",
    index=False,
)


# ============================================================
# ACTION SUMMARY
# ============================================================

def action_summary(df):

    rows = []

    for action, g in df.groupby(
        "recommended_action",
        dropna=False,
    ):

        rows.append({
            "recommended_action": action,
            "n": len(g),
            "fraction": len(g) / len(df),
            "mean_priority": safe_mean(
                g["review_priority"]
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "mean_priority",
        ascending=False,
    )


val_actions = action_summary(val_e)
test_actions = action_summary(test_e)

val_actions.to_csv(
    OUT_DIR / "validation_action_summary.csv",
    index=False,
)

test_actions.to_csv(
    OUT_DIR / "test_action_summary.csv",
    index=False,
)


# ============================================================
# POST-HOC TEST METRICS
# ============================================================
#
# Labels are used HERE ONLY for evaluation.
#
# They do not influence normalization, weights or thresholds.
# ============================================================

print()
print("=" * 70)
print("POST-HOC TEST EVALUATION")
print("=" * 70)

metrics = []


score_columns = {
    "YOLO": "yolo_evidence",
    "VAE": "vae_evidence",
    "FLOW": "flow_evidence",
    "YOLO_VAE_FLOW_PRIORITY": "review_priority",
    "BASE_EVIDENCE": "base_evidence_score",
}


for name, col in score_columns.items():

    m = binary_metrics(
        test_e["label"].values,
        test_e[col].values,
    )

    m["score"] = name

    metrics.append(m)

    print(
        f"{name:28s} "
        f"ROC={m['roc_auc']:.6f} "
        f"PR={m['pr_auc']:.6f} "
        f"BestF1={m['best_f1']:.6f}"
    )


metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    OUT_DIR / "posthoc_test_metrics.csv",
    index=False,
)


# ============================================================
# DATASET-WISE TEST METRICS
# ============================================================

domain_rows = []

for dataset, g in test_e.groupby(
    "dataset",
    dropna=False,
):

    for name, col in score_columns.items():

        m = binary_metrics(
            g["label"].values,
            g[col].values,
        )

        domain_rows.append({
            "dataset": dataset,
            "score": name,
            **m,
        })


domain_df = pd.DataFrame(domain_rows)

domain_df.to_csv(
    OUT_DIR / "dataset_wise_test_metrics.csv",
    index=False,
)


# ============================================================
# TOP REVIEW QUEUE
# ============================================================

TOP_N = min(100, len(test_e))

queue_cols = [
    "rank",
    "output_image",
    "path",
    "dataset",
    "group",
    "label",
    "yolo_raw",
    "vae_raw",
    "flow_raw",
    "yolo_evidence",
    "vae_evidence",
    "flow_evidence",
    "base_evidence_score",
    "corroboration",
    "review_priority",
    "yolo_band",
    "vae_band",
    "flow_band",
    "evidence_profile",
    "recommended_action",
]

test_e.head(TOP_N)[queue_cols].to_csv(
    OUT_DIR / "top_review_queue.csv",
    index=False,
)


# ============================================================
# MACHINE-READABLE CONFIG
# ============================================================

config = {
    "engine": "SAAD Evidence Engine v2",

    "normalization": {
        "reference": "validation_normals_only",
        "low_percentile": LOW_PCT,
        "high_percentile": HIGH_PCT,
    },

    "weights": {
        "yolo": W_YOLO,
        "vae": W_VAE,
        "flow": W_FLOW,
    },

    "corroboration_bonus": CORROBORATION_BONUS,
    "low_evidence_penalty": LOW_EVIDENCE_PENALTY,

    "interpretation": {
        "score_type": "review_priority",
        "probability": False,
        "calibrated_probability": False,
        "test_labels_used_for_fitting": False,
    },

    "inputs": {
        "validation": str(VAL_CSV),
        "test": str(TEST_CSV),
    },
}

with open(
    OUT_DIR / "engine_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_lines = []

report_lines.append(
    "SAAD EVIDENCE ENGINE v2\n"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    "\nPURPOSE\n"
    "Convert frozen YOLO + VAE + Flow evidence into an\n"
    "interpretable review-priority ranking.\n"
)

report_lines.append(
    "\nIMPORTANT INTERPRETATION\n"
    "- review_priority is NOT a probability.\n"
    "- No test labels were used for fitting.\n"
    "- Normalization uses validation normals only.\n"
    "- YOLO is treated as primary detector evidence.\n"
    "- VAE and Flow provide normality corroboration.\n"
)

report_lines.append(
    "\nWEIGHTS\n"
    f"YOLO = {W_YOLO:.2f}\n"
    f"VAE  = {W_VAE:.2f}\n"
    f"Flow = {W_FLOW:.2f}\n"
)

report_lines.append(
    "\nDATASET SIZES\n"
    f"Validation = {len(val_e)}\n"
    f"Validation normals = {len(val_norm)}\n"
    f"Test = {len(test_e)}\n"
)

report_lines.append(
    "\nPOST-HOC TEST METRICS\n"
)

for _, row in metrics_df.iterrows():

    report_lines.append(
        f"{row['score']}: "
        f"ROC-AUC={row['roc_auc']:.6f}, "
        f"PR-AUC={row['pr_auc']:.6f}, "
        f"Best-F1={row['best_f1']:.6f}\n"
    )


report_lines.append(
    "\nVALIDATION ACTION DISTRIBUTION\n"
)

for _, row in val_actions.iterrows():

    report_lines.append(
        f"{row['recommended_action']}: "
        f"{int(row['n'])} "
        f"({row['fraction'] * 100:.2f}%)\n"
    )


report_lines.append(
    "\nTEST ACTION DISTRIBUTION\n"
)

for _, row in test_actions.iterrows():

    report_lines.append(
        f"{row['recommended_action']}: "
        f"{int(row['n'])} "
        f"({row['fraction'] * 100:.2f}%)\n"
    )


report_lines.append(
    "\nEVIDENCE PROFILES — TEST\n"
)

for _, row in test_profile.iterrows():

    report_lines.append(
        f"{row['evidence_profile']}: "
        f"{int(row['n'])} "
        f"mean_priority={row['mean_priority']:.4f}\n"
    )


report_lines.append(
    "\nOUTPUT FILES\n"
    "normalization_parameters.csv\n"
    "validation_evidence_queue.csv\n"
    "test_evidence_queue.csv\n"
    "validation_profile_summary.csv\n"
    "test_profile_summary.csv\n"
    "validation_action_summary.csv\n"
    "test_action_summary.csv\n"
    "posthoc_test_metrics.csv\n"
    "dataset_wise_test_metrics.csv\n"
    "top_review_queue.csv\n"
    "engine_config.json\n"
)

with open(
    OUT_DIR / "EVIDENCE_ENGINE_REPORT.txt",
    "w",
    encoding="utf-8",
) as f:

    f.writelines(report_lines)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("ENGINE COMPLETE")
print("=" * 70)

print()
print("Output directory:")
print(OUT_DIR)

print()
print("Test action distribution:")

for _, row in test_actions.iterrows():

    print(
        f"  {row['recommended_action']:22s} "
        f"{int(row['n']):4d} "
        f"({row['fraction'] * 100:6.2f}%)"
    )

print()
print("Top 10 test candidates:")

display_cols = [
    "rank",
    "output_image",
    "dataset",
    "review_priority",
    "evidence_profile",
    "recommended_action",
]

print(
    test_e.head(10)[display_cols].to_string(
        index=False
    )
)

print()
print("Done.")