# ============================================================
# SAAD — LEAVE-ONE-DOMAIN-OUT CALIBRATION / FUSION
# ============================================================
#
# Purpose:
#   Test whether SAAD calibration/fusion remains effective
#   when an entire sonar dataset/domain is held out.
#
# IMPORTANT:
#   - Uses ONLY validation_raw_scores.csv
#   - Does NOT touch the final test set
#   - No model retraining
#   - No checkpoint modification
#
# Domains:
#   GhostVision
#   AI4Shipwrecks
#   SubPipeMini2
#
# For each held-out domain:
#
#   TRAIN/TUNE:
#       remaining two domains
#
#   HOLDOUT:
#       one entire domain
#
# Calibration:
#   1. Percentile
#   2. Platt
#   3. Isotonic
#
# Fusion:
#   1. Equal weights
#   2. Validation-optimized weights
#
# Weight search:
#   non-negative weights summing to 1
#   step = 0.05
#
# ============================================================

import os
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = (
    r"E:\SIH\SIH_Results\validation_weighted_fusion"
    r"\validation_raw_scores.csv"
)

OUTPUT_DIR = (
    r"E:\SIH\SIH_Results"
    r"\calibration_leave_one_domain_out"
)

WEIGHT_STEP = 0.05

RANDOM_STATE = 42


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_metric(metric_fn, y_true, scores):
    """
    Return NaN instead of crashing when a held-out domain
    contains only one class.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(metric_fn(y_true, scores))
    except Exception:
        return np.nan


def best_f1(y_true, scores):
    """
    Find threshold maximizing F1.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores
    )

    if len(thresholds) == 0:
        return np.nan, np.nan, np.nan, np.nan

    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    idx = int(np.nanargmax(f1))

    threshold = float(thresholds[idx])
    p = float(precision[idx])
    r = float(recall[idx])
    f = float(f1[idx])

    return threshold, p, r, f


def evaluate_scores(y_true, scores):
    """
    Return ROC-AUC, PR-AUC and Best-F1 metrics.
    """
    roc = safe_metric(roc_auc_score, y_true, scores)
    pr = safe_metric(average_precision_score, y_true, scores)

    threshold, precision, recall, f1 = best_f1(
        y_true,
        scores
    )

    return {
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "Best_F1": f1,
        "Best_F1_precision": precision,
        "Best_F1_recall": recall,
        "Best_F1_threshold": threshold,
    }


# ============================================================
# CALIBRATION
# ============================================================

def percentile_fit(train_scores, train_labels):
    """
    Fit percentile normalization using NORMAL samples only.

    NORMAL = label 0
    """
    normal = train_scores[train_labels == 0]

    if len(normal) < 2:
        p5 = float(np.min(train_scores))
        p95 = float(np.max(train_scores))
    else:
        p5 = float(np.percentile(normal, 5))
        p95 = float(np.percentile(normal, 95))

    if p95 <= p5:
        p95 = p5 + 1e-8

    return {
        "p5": p5,
        "p95": p95,
    }


def percentile_transform(scores, params):
    """
    Map scores approximately into [0,1].

    Higher = more anomalous.
    """
    p5 = params["p5"]
    p95 = params["p95"]

    x = (scores - p5) / (p95 - p5 + 1e-12)

    return np.clip(x, 0.0, 1.0)


def platt_fit(train_scores, train_labels):
    """
    Fit logistic calibration.

    class_weight='balanced' is important because the validation
    data is heavily imbalanced.
    """
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )

    model.fit(
        train_scores.reshape(-1, 1),
        train_labels,
    )

    return model


def platt_transform(scores, model):
    """
    Probability of anthropogenic class.
    """
    return model.predict_proba(
        scores.reshape(-1, 1)
    )[:, 1]


def isotonic_fit(train_scores, train_labels):
    """
    Fit monotonic calibration.
    """
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(
        train_scores,
        train_labels,
    )

    return model


def isotonic_transform(scores, model):
    return model.predict(scores)


# ============================================================
# FUSION
# ============================================================

def generate_weights(step=0.05):
    """
    Generate all non-negative 3-way weights summing to 1.
    """
    weights = []

    n = int(round(1.0 / step))

    for i in range(n + 1):
        for j in range(n + 1):
            k = n - i - j

            if k < 0:
                continue

            w1 = i * step
            w2 = j * step
            w3 = k * step

            weights.append(
                (
                    round(w1, 10),
                    round(w2, 10),
                    round(w3, 10),
                )
            )

    return weights


WEIGHTS = generate_weights(WEIGHT_STEP)


def fuse(yolo, vae, flow, weights):
    wy, wv, wf = weights

    return (
        wy * yolo
        + wv * vae
        + wf * flow
    )


def optimize_weights(
    yolo,
    vae,
    flow,
    labels,
):
    """
    Select weights using ONLY the training portion
    of the LODO experiment.

    Primary objective:
        PR-AUC

    Tie-break:
        ROC-AUC
    """

    best = None

    for weights in WEIGHTS:

        scores = fuse(
            yolo,
            vae,
            flow,
            weights,
        )

        pr = safe_metric(
            average_precision_score,
            labels,
            scores,
        )

        roc = safe_metric(
            roc_auc_score,
            labels,
            scores,
        )

        f1_threshold, f1_p, f1_r, f1 = best_f1(
            labels,
            scores,
        )

        if np.isnan(pr):
            continue

        candidate = {
            "w_yolo": weights[0],
            "w_vae": weights[1],
            "w_flow": weights[2],
            "PR_AUC": pr,
            "ROC_AUC": roc,
            "Best_F1": f1,
            "Best_F1_precision": f1_p,
            "Best_F1_recall": f1_r,
            "Best_F1_threshold": f1_threshold,
        }

        if best is None:
            best = candidate
        else:

            better_pr = candidate["PR_AUC"] > best["PR_AUC"] + 1e-12

            equal_pr = abs(
                candidate["PR_AUC"] - best["PR_AUC"]
            ) <= 1e-12

            better_roc = (
                equal_pr
                and candidate["ROC_AUC"] > best["ROC_AUC"]
            )

            if better_pr or better_roc:
                best = candidate

    return best


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAAD — LEAVE-ONE-DOMAIN-OUT CALIBRATION / FUSION")
print("=" * 70)

print()
print("Loading validation scores...")
print(INPUT_CSV)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print()
print(f"Validation images: {len(df)}")

# ------------------------------------------------------------
# Detect expected columns
# ------------------------------------------------------------

required_candidates = {
    "label": ["label", "y_true", "target", "class"],
    "dataset": ["dataset", "domain", "source_dataset"],
    "yolo": ["yolo_raw", "yolo_score", "YOLO", "yolo"],
    "vae": ["vae_raw", "vae_score", "VAE", "vae"],
    "flow": ["flow_raw", "flow_score", "Flow", "flow"],
}


def find_column(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


LABEL_COL = find_column(required_candidates["label"])
DATASET_COL = find_column(required_candidates["dataset"])
YOLO_COL = find_column(required_candidates["yolo"])
VAE_COL = find_column(required_candidates["vae"])
FLOW_COL = find_column(required_candidates["flow"])


missing = []

if LABEL_COL is None:
    missing.append(
        "label / y_true / target / class"
    )

if DATASET_COL is None:
    missing.append(
        "dataset / domain / source_dataset"
    )

if YOLO_COL is None:
    missing.append(
        "yolo_raw / yolo_score / YOLO / yolo"
    )

if VAE_COL is None:
    missing.append(
        "vae_raw / vae_score / VAE / vae"
    )

if FLOW_COL is None:
    missing.append(
        "flow_raw / flow_score / Flow / flow"
    )

if missing:
    print()
    print("ERROR: Could not identify required columns:")
    for x in missing:
        print("  -", x)

    print()
    print("Available columns:")
    for c in df.columns:
        print(" ", c)

    raise RuntimeError(
        "Required columns missing."
    )


print()
print("Detected columns:")
print("  Dataset :", DATASET_COL)
print("  Label   :", LABEL_COL)
print("  YOLO    :", YOLO_COL)
print("  VAE     :", VAE_COL)
print("  Flow    :", FLOW_COL)


# ============================================================
# CLEAN DATA
# ============================================================

work = df[
    [
        DATASET_COL,
        LABEL_COL,
        YOLO_COL,
        VAE_COL,
        FLOW_COL,
    ]
].copy()

work.columns = [
    "dataset",
    "label",
    "yolo",
    "vae",
    "flow",
]

work["label"] = work["label"].astype(int)

for c in ["yolo", "vae", "flow"]:
    work[c] = pd.to_numeric(
        work[c],
        errors="coerce",
    )

before = len(work)

work = work.dropna(
    subset=[
        "dataset",
        "label",
        "yolo",
        "vae",
        "flow",
    ]
).reset_index(drop=True)

after = len(work)

if before != after:
    print()
    print(
        f"Removed {before - after} rows containing NaN."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOMAIN DISTRIBUTION")
print("=" * 70)

domain_summary = (
    work.groupby("dataset")["label"]
    .agg(
        images="count",
        normal=lambda x: int((x == 0).sum()),
        anthropogenic=lambda x: int((x == 1).sum()),
    )
    .reset_index()
)

print(domain_summary.to_string(index=False))


domains = sorted(
    work["dataset"].unique().tolist()
)

print()
print("Domains detected:")

for d in domains:
    print(" ", d)

if len(domains) < 2:
    raise RuntimeError(
        "Need at least two domains for LODO."
    )


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

ensure_dir(OUTPUT_DIR)

ensure_dir(
    os.path.join(
        OUTPUT_DIR,
        "fold_details",
    )
)


# ============================================================
# EQUAL WEIGHTS
# ============================================================

EQUAL_WEIGHTS = (
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
)


# ============================================================
# MAIN LODO LOOP
# ============================================================

all_results = []
weight_results = []
calibration_details = []

print()
print("=" * 70)
print("RUNNING LEAVE-ONE-DOMAIN-OUT")
print("=" * 70)

for fold_id, held_out_domain in enumerate(domains, start=1):

    print()
    print("-" * 70)
    print(
        f"FOLD {fold_id}/{len(domains)}"
    )
    print(
        f"Held-out domain: {held_out_domain}"
    )
    print("-" * 70)

    train_df = work[
        work["dataset"] != held_out_domain
    ].copy()

    holdout_df = work[
        work["dataset"] == held_out_domain
    ].copy()

    print(
        f"Train/tune: {len(train_df)}"
    )

    print(
        f"Holdout   : {len(holdout_df)}"
    )

    print()
    print("Train class counts:")

    print(
        train_df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Holdout class counts:")

    print(
        holdout_df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # --------------------------------------------------------
    # Extract arrays
    # --------------------------------------------------------

    y_train = train_df["label"].values

    y_holdout = holdout_df["label"].values

    train_yolo = train_df["yolo"].values.astype(float)
    train_vae = train_df["vae"].values.astype(float)
    train_flow = train_df["flow"].values.astype(float)

    hold_yolo = holdout_df["yolo"].values.astype(float)
    hold_vae = holdout_df["vae"].values.astype(float)
    hold_flow = holdout_df["flow"].values.astype(float)


    # ========================================================
    # CALIBRATION
    # ========================================================

    calibrators = {}

    # --------------------------------------------------------
    # Percentile
    # --------------------------------------------------------

    pct_yolo_params = percentile_fit(
        train_yolo,
        y_train,
    )

    pct_vae_params = percentile_fit(
        train_vae,
        y_train,
    )

    pct_flow_params = percentile_fit(
        train_flow,
        y_train,
    )

    train_pct_yolo = percentile_transform(
        train_yolo,
        pct_yolo_params,
    )

    train_pct_vae = percentile_transform(
        train_vae,
        pct_vae_params,
    )

    train_pct_flow = percentile_transform(
        train_flow,
        pct_flow_params,
    )

    hold_pct_yolo = percentile_transform(
        hold_yolo,
        pct_yolo_params,
    )

    hold_pct_vae = percentile_transform(
        hold_vae,
        pct_vae_params,
    )

    hold_pct_flow = percentile_transform(
        hold_flow,
        pct_flow_params,
    )

    calibrators["percentile"] = {
        "train": (
            train_pct_yolo,
            train_pct_vae,
            train_pct_flow,
        ),
        "holdout": (
            hold_pct_yolo,
            hold_pct_vae,
            hold_pct_flow,
        ),
    }


    # --------------------------------------------------------
    # Platt
    # --------------------------------------------------------

    platt_yolo = platt_fit(
        train_yolo,
        y_train,
    )

    platt_vae = platt_fit(
        train_vae,
        y_train,
    )

    platt_flow = platt_fit(
        train_flow,
        y_train,
    )

    train_platt_yolo = platt_transform(
        train_yolo,
        platt_yolo,
    )

    train_platt_vae = platt_transform(
        train_vae,
        platt_vae,
    )

    train_platt_flow = platt_transform(
        train_flow,
        platt_flow,
    )

    hold_platt_yolo = platt_transform(
        hold_yolo,
        platt_yolo,
    )

    hold_platt_vae = platt_transform(
        hold_vae,
        platt_vae,
    )

    hold_platt_flow = platt_transform(
        hold_flow,
        platt_flow,
    )

    calibrators["platt"] = {
        "train": (
            train_platt_yolo,
            train_platt_vae,
            train_platt_flow,
        ),
        "holdout": (
            hold_platt_yolo,
            hold_platt_vae,
            hold_platt_flow,
        ),
    }


    # --------------------------------------------------------
    # Isotonic
    # --------------------------------------------------------

    iso_yolo = isotonic_fit(
        train_yolo,
        y_train,
    )

    iso_vae = isotonic_fit(
        train_vae,
        y_train,
    )

    iso_flow = isotonic_fit(
        train_flow,
        y_train,
    )

    train_iso_yolo = isotonic_transform(
        train_yolo,
        iso_yolo,
    )

    train_iso_vae = isotonic_transform(
        train_vae,
        iso_vae,
    )

    train_iso_flow = isotonic_transform(
        train_flow,
        iso_flow,
    )

    hold_iso_yolo = isotonic_transform(
        hold_yolo,
        iso_yolo,
    )

    hold_iso_vae = isotonic_transform(
        hold_vae,
        iso_vae,
    )

    hold_iso_flow = isotonic_transform(
        hold_flow,
        iso_flow,
    )

    calibrators["isotonic"] = {
        "train": (
            train_iso_yolo,
            train_iso_vae,
            train_iso_flow,
        ),
        "holdout": (
            hold_iso_yolo,
            hold_iso_vae,
            hold_iso_flow,
        ),
    }


    # ========================================================
    # EACH CALIBRATION METHOD
    # ========================================================

    for calibration_name in [
        "percentile",
        "platt",
        "isotonic",
    ]:

        train_scores = calibrators[
            calibration_name
        ]["train"]

        hold_scores = calibrators[
            calibration_name
        ]["holdout"]


        train_yolo_c = train_scores[0]
        train_vae_c = train_scores[1]
        train_flow_c = train_scores[2]

        hold_yolo_c = hold_scores[0]
        hold_vae_c = hold_scores[1]
        hold_flow_c = hold_scores[2]


        # ====================================================
        # EQUAL FUSION
        # ====================================================

        equal_hold = fuse(
            hold_yolo_c,
            hold_vae_c,
            hold_flow_c,
            EQUAL_WEIGHTS,
        )

        equal_metrics = evaluate_scores(
            y_holdout,
            equal_hold,
        )

        all_results.append({
            "fold": fold_id,
            "held_out_domain": held_out_domain,
            "calibration": calibration_name,
            "fusion": "equal",

            "w_yolo": EQUAL_WEIGHTS[0],
            "w_vae": EQUAL_WEIGHTS[1],
            "w_flow": EQUAL_WEIGHTS[2],

            "train_images": len(train_df),
            "holdout_images": len(holdout_df),

            "train_normal": int(
                (y_train == 0).sum()
            ),
            "train_anthropogenic": int(
                (y_train == 1).sum()
            ),

            "holdout_normal": int(
                (y_holdout == 0).sum()
            ),
            "holdout_anthropogenic": int(
                (y_holdout == 1).sum()
            ),

            **{
                f"holdout_{k}": v
                for k, v in equal_metrics.items()
            },
        })


        # ====================================================
        # OPTIMIZED FUSION
        # ====================================================

        best = optimize_weights(
            train_yolo_c,
            train_vae_c,
            train_flow_c,
            y_train,
        )

        if best is None:
            print(
                f"WARNING: could not optimize "
                f"{calibration_name}"
            )
            continue

        best_weights = (
            best["w_yolo"],
            best["w_vae"],
            best["w_flow"],
        )

        optimized_hold = fuse(
            hold_yolo_c,
            hold_vae_c,
            hold_flow_c,
            best_weights,
        )

        optimized_metrics = evaluate_scores(
            y_holdout,
            optimized_hold,
        )


        all_results.append({
            "fold": fold_id,
            "held_out_domain": held_out_domain,
            "calibration": calibration_name,
            "fusion": "optimized",

            "w_yolo": best_weights[0],
            "w_vae": best_weights[1],
            "w_flow": best_weights[2],

            "train_images": len(train_df),
            "holdout_images": len(holdout_df),

            "train_normal": int(
                (y_train == 0).sum()
            ),
            "train_anthropogenic": int(
                (y_train == 1).sum()
            ),

            "holdout_normal": int(
                (y_holdout == 0).sum()
            ),
            "holdout_anthropogenic": int(
                (y_holdout == 1).sum()
            ),

            **{
                f"holdout_{k}": v
                for k, v in optimized_metrics.items()
            },
        })


        # ----------------------------------------------------
        # Weight stability record
        # ----------------------------------------------------

        weight_results.append({
            "fold": fold_id,
            "held_out_domain": held_out_domain,
            "calibration": calibration_name,

            "w_yolo": best_weights[0],
            "w_vae": best_weights[1],
            "w_flow": best_weights[2],

            "train_PR_AUC": best["PR_AUC"],
            "train_ROC_AUC": best["ROC_AUC"],
            "train_Best_F1": best["Best_F1"],
        })


        # ----------------------------------------------------
        # Calibration parameter summary
        # ----------------------------------------------------

        calibration_details.append({
            "fold": fold_id,
            "held_out_domain": held_out_domain,
            "calibration": calibration_name,

            "train_yolo_mean": float(
                np.mean(train_yolo_c)
            ),
            "train_vae_mean": float(
                np.mean(train_vae_c)
            ),
            "train_flow_mean": float(
                np.mean(train_flow_c)
            ),

            "holdout_yolo_mean": float(
                np.mean(hold_yolo_c)
            ),
            "holdout_vae_mean": float(
                np.mean(hold_vae_c)
            ),
            "holdout_flow_mean": float(
                np.mean(hold_flow_c)
            ),
        })


        print()
        print(
            f"{calibration_name.upper()}"
        )

        print(
            "  Optimized weights:"
            f" YOLO={best_weights[0]:.2f}"
            f" VAE={best_weights[1]:.2f}"
            f" Flow={best_weights[2]:.2f}"
        )

        print(
            f"  Holdout ROC-AUC:"
            f" {optimized_metrics['ROC_AUC']:.4f}"
        )

        print(
            f"  Holdout PR-AUC:"
            f" {optimized_metrics['PR_AUC']:.4f}"
        )

        print(
            f"  Holdout Best-F1:"
            f" {optimized_metrics['Best_F1']:.4f}"
        )


    # ========================================================
    # SAVE PER-FOLD DETAILS
    # ========================================================

    fold_path = os.path.join(
        OUTPUT_DIR,
        "fold_details",
        f"fold_{fold_id}_{held_out_domain}.csv",
    )

    fold_rows = [
        r
        for r in all_results
        if r["fold"] == fold_id
    ]

    pd.DataFrame(fold_rows).to_csv(
        fold_path,
        index=False,
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

weights_df = pd.DataFrame(
    weight_results
)

calibration_df = pd.DataFrame(
    calibration_details
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_fold_results.csv",
    ),
    index=False,
)

weights_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_weight_stability.csv",
    ),
    index=False,
)

calibration_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_calibration_details.csv",
    ),
    index=False,
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        [
            "calibration",
            "fusion",
        ],
        dropna=False,
    )
    .agg(
        holdout_ROC_AUC_mean=(
            "holdout_ROC_AUC",
            "mean",
        ),
        holdout_ROC_AUC_std=(
            "holdout_ROC_AUC",
            "std",
        ),

        holdout_PR_AUC_mean=(
            "holdout_PR_AUC",
            "mean",
        ),
        holdout_PR_AUC_std=(
            "holdout_PR_AUC",
            "std",
        ),

        holdout_Best_F1_mean=(
            "holdout_Best_F1",
            "mean",
        ),
        holdout_Best_F1_std=(
            "holdout_Best_F1",
            "std",
        ),
    )
    .reset_index()
)

summary = summary.sort_values(
    "holdout_PR_AUC_mean",
    ascending=False,
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_performance_summary.csv",
    ),
    index=False,
)


# ============================================================
# DOMAIN-BY-METHOD TABLE
# ============================================================

domain_summary = (
    results_df[
        results_df["fusion"] == "optimized"
    ]
    .pivot_table(
        index="held_out_domain",
        columns="calibration",
        values=[
            "holdout_ROC_AUC",
            "holdout_PR_AUC",
            "holdout_Best_F1",
        ],
        aggfunc="first",
    )
)

domain_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_domain_summary.csv",
    )
)


# ============================================================
# WEIGHT SUMMARY
# ============================================================

weight_summary = (
    weights_df
    .groupby("calibration")
    .agg(
        w_yolo_mean=("w_yolo", "mean"),
        w_yolo_std=("w_yolo", "std"),
        w_yolo_min=("w_yolo", "min"),
        w_yolo_max=("w_yolo", "max"),

        w_vae_mean=("w_vae", "mean"),
        w_vae_std=("w_vae", "std"),
        w_vae_min=("w_vae", "min"),
        w_vae_max=("w_vae", "max"),

        w_flow_mean=("w_flow", "mean"),
        w_flow_std=("w_flow", "std"),
        w_flow_min=("w_flow", "min"),
        w_flow_max=("w_flow", "max"),
    )
    .reset_index()
)

weight_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lodo_weight_summary.csv",
    ),
    index=False,
)


# ============================================================
# TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "lodo_summary.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SAAD — LEAVE-ONE-DOMAIN-OUT "
        "CALIBRATION / FUSION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "IMPORTANT:\n"
        "This experiment uses ONLY the validation set.\n"
        "The final held-out test set was NOT accessed.\n\n"
    )

    f.write(
        f"Validation images: {len(work)}\n"
    )

    f.write(
        f"Domains: {', '.join(domains)}\n"
    )

    f.write(
        f"Weight combinations: {len(WEIGHTS)}\n\n"
    )

    f.write(
        "DOMAIN DISTRIBUTION\n"
    )

    f.write(
        domain_summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "PERFORMANCE SUMMARY\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "WEIGHT SUMMARY\n"
    )

    f.write(
        weight_summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "INTERPRETATION GUIDANCE\n"
    )

    f.write(
        "1. Prefer methods with good held-out-domain "
        "PR-AUC and low variance.\n"
    )

    f.write(
        "2. Stable weights are more trustworthy than "
        "a single validation optimum.\n"
    )

    f.write(
        "3. Large domain-to-domain degradation indicates "
        "domain shift.\n"
    )

    f.write(
        "4. Do NOT use the final test set to choose "
        "calibration or weights.\n"
    )

    f.write(
        "5. Because SubPipeMini2 may contain only "
        "positive samples in some splits, ROC-AUC "
        "may be undefined there.\n"
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("LODO PERFORMANCE SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("LODO WEIGHT SUMMARY")
print("=" * 70)

print(
    weight_summary.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("DOMAIN-BY-DOMAIN OPTIMIZED RESULTS")
print("=" * 70)

optimized = results_df[
    results_df["fusion"] == "optimized"
].copy()

for domain in domains:

    print()
    print(
        f"HELD OUT: {domain}"
    )

    temp = optimized[
        optimized["held_out_domain"] == domain
    ][
        [
            "calibration",
            "w_yolo",
            "w_vae",
            "w_flow",
            "holdout_ROC_AUC",
            "holdout_PR_AUC",
            "holdout_Best_F1",
        ]
    ]

    print(
        temp.to_string(
            index=False
        )
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("Results saved to:")
print(OUTPUT_DIR)

print()
print("Important files:")

print(
    "  lodo_fold_results.csv"
)

print(
    "  lodo_weight_stability.csv"
)

print(
    "  lodo_weight_summary.csv"
)

print(
    "  lodo_performance_summary.csv"
)

print(
    "  lodo_domain_summary.csv"
)

print(
    "  lodo_calibration_details.csv"
)

print(
    "  lodo_summary.txt"
)