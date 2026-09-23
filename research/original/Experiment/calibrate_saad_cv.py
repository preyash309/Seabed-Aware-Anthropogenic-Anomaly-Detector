# ============================================================
# SAAD — 5-FOLD CALIBRATION / FUSION STABILITY EXPERIMENT
# ============================================================
#
# PURPOSE
# -------
# Determine whether score calibration and fusion weights are
# stable across different subsets of the validation data.
#
# Signals:
#   YOLO
#   VAE
#   Flow
#
# Calibration:
#   1. Percentile normalization
#   2. Platt / logistic calibration
#   3. Isotonic regression
#
# Fusion:
#   - Equal weight
#   - Validation-fold optimized weights
#
# IMPORTANT:
#   * TEST SET IS NEVER USED.
#   * Calibration is fitted inside each training fold.
#   * Fusion weights are selected inside each training fold.
#   * Held-out fold is evaluated only after calibration/weights
#     have been fitted.
#
# This produces:
#   - fold_results.csv
#   - weight_stability.csv
#   - oof_results.csv
#   - calibration_cv_summary.txt
#
# ============================================================


import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_DIR = Path(
    r"E:\SIH\SIH_Results\validation_weighted_fusion"
)

OUTPUT_DIR = Path(
    r"E:\SIH\SIH_Results\calibration_cross_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_FILE = (
    INPUT_DIR
    / "validation_raw_scores.csv"
)

N_SPLITS = 5

WEIGHT_STEP = 0.05


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD VALIDATION SCORES
# ============================================================

print("=" * 70)
print("SAAD — 5-FOLD CALIBRATION / FUSION STABILITY")
print("=" * 70)

print("\nLoading validation scores...")

if not VAL_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n{VAL_FILE}"
    )

df = pd.read_csv(
    VAL_FILE
)

print(
    "Validation images:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "label",
    "yolo_raw",
    "vae_raw",
    "flow_raw",
]

for col in required_columns:

    if col not in df.columns:

        raise RuntimeError(
            f"Missing required column: {col}"
        )


# ============================================================
# CLEAN
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required_columns
).reset_index(
    drop=True
)


# ============================================================
# ARRAYS
# ============================================================

y = df[
    "label"
].astype(int).to_numpy()

print("\nClass distribution:")

print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)


if len(
    np.unique(y)
) != 2:

    raise RuntimeError(
        "Validation set must contain both classes."
    )


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

SIGNALS = [
    "yolo",
    "vae",
    "flow",
]

RAW_COLUMNS = {
    "yolo": "yolo_raw",
    "vae": "vae_raw",
    "flow": "flow_raw",
}


# ============================================================
# WEIGHT GRID
# ============================================================

def generate_weights(
    step=0.05
):

    weights = []

    n = int(
        round(
            1.0 / step
        )
    )

    for iy in range(
        n + 1
    ):

        wy = iy * step

        for iv in range(
            n + 1
        ):

            wv = iv * step

            wf = (
                1.0
                - wy
                - wv
            )

            if wf < -1e-9:
                continue

            wf = max(
                0.0,
                wf
            )

            weights.append(
                (
                    round(
                        wy,
                        6
                    ),
                    round(
                        wv,
                        6
                    ),
                    round(
                        wf,
                        6
                    ),
                )
            )

    return weights


WEIGHT_GRID = generate_weights(
    WEIGHT_STEP
)

print(
    "\nWeight combinations:",
    len(WEIGHT_GRID)
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_scores(
    y_true,
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    roc = roc_auc_score(
        y_true,
        scores
    )

    pr = average_precision_score(
        y_true,
        scores
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1)
    )

    return {
        "ROC_AUC":
            float(roc),

        "PR_AUC":
            float(pr),

        "Best_F1":
            float(
                f1[best_idx]
            ),

        "Best_F1_precision":
            float(
                precision[best_idx]
            ),

        "Best_F1_recall":
            float(
                recall[best_idx]
            ),
    }


# ============================================================
# CALIBRATION FITTING
# ============================================================

def fit_percentile(
    train_df
):

    normal = train_df[
        train_df["label"] == 0
    ]

    stats = {}

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        p5 = float(
            np.percentile(
                normal[
                    raw_col
                ],
                5
            )
        )

        p95 = float(
            np.percentile(
                normal[
                    raw_col
                ],
                95
            )
        )

        if p95 <= p5:

            p95 = p5 + 1e-8

        stats[
            signal
        ] = (
            p5,
            p95
        )

    return stats


def apply_percentile(
    data_df,
    stats
):

    output = {}

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        p5, p95 = stats[
            signal
        ]

        values = (
            data_df[
                raw_col
            ].to_numpy(
                dtype=np.float64
            )
        )

        z = (
            values - p5
        ) / (
            p95 - p5
        )

        output[
            signal
        ] = np.clip(
            z,
            0.0,
            1.0
        )

    return output


# ------------------------------------------------------------
# Platt
# ------------------------------------------------------------

def fit_platt(
    train_df
):

    models = {}

    train_y = train_df[
        "label"
    ].astype(int).to_numpy()

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        x = train_df[
            [raw_col]
        ].to_numpy(
            dtype=np.float64
        )

        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )

        model.fit(
            x,
            train_y
        )

        models[
            signal
        ] = model

    return models


def apply_platt(
    data_df,
    models
):

    output = {}

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        x = data_df[
            [raw_col]
        ].to_numpy(
            dtype=np.float64
        )

        output[
            signal
        ] = models[
            signal
        ].predict_proba(
            x
        )[:, 1]

    return output


# ------------------------------------------------------------
# Isotonic
# ------------------------------------------------------------

def fit_isotonic(
    train_df
):

    models = {}

    train_y = train_df[
        "label"
    ].astype(int).to_numpy()

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        x = train_df[
            raw_col
        ].to_numpy(
            dtype=np.float64
        )

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )

        model.fit(
            x,
            train_y
        )

        models[
            signal
        ] = model

    return models


def apply_isotonic(
    data_df,
    models
):

    output = {}

    for signal in SIGNALS:

        raw_col = RAW_COLUMNS[
            signal
        ]

        x = data_df[
            raw_col
        ].to_numpy(
            dtype=np.float64
        )

        output[
            signal
        ] = models[
            signal
        ].predict(
            x
        )

    return output


# ============================================================
# FUSION
# ============================================================

def fuse(
    calibrated,
    wy,
    wv,
    wf
):

    return (
        wy * calibrated["yolo"]
        + wv * calibrated["vae"]
        + wf * calibrated["flow"]
    )


# ============================================================
# SEARCH WEIGHTS
# ============================================================

def optimize_weights(
    calibrated,
    labels
):

    best = None

    all_results = []

    for wy, wv, wf in WEIGHT_GRID:

        scores = fuse(
            calibrated,
            wy,
            wv,
            wf
        )

        pr = average_precision_score(
            labels,
            scores
        )

        roc = roc_auc_score(
            labels,
            scores
        )

        precision, recall, thresholds = (
            precision_recall_curve(
                labels,
                scores
            )
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
                + 1e-12
            )
        )

        best_f1 = float(
            np.max(f1)
        )

        row = {

            "w_yolo":
                wy,

            "w_vae":
                wv,

            "w_flow":
                wf,

            "ROC_AUC":
                roc,

            "PR_AUC":
                pr,

            "Best_F1":
                best_f1,
        }

        all_results.append(
            row
        )

        # PRIMARY OBJECTIVE:
        # PR-AUC
        #
        # Secondary:
        # ROC-AUC
        # Secondary:
        # Best F1

        key = (
            pr,
            roc,
            best_f1
        )

        if (
            best is None
            or key > best["key"]
        ):

            best = {
                "key": key,
                "row": row,
            }

    return (
        best["row"],
        pd.DataFrame(
            all_results
        )
    )


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


# ============================================================
# STORAGE
# ============================================================

fold_results = []

weight_rows = []

oof_storage = {

    "percentile": np.full(
        len(df),
        np.nan,
        dtype=np.float64
    ),

    "platt": np.full(
        len(df),
        np.nan,
        dtype=np.float64
    ),

    "isotonic": np.full(
        len(df),
        np.nan,
        dtype=np.float64
    ),
}


# ============================================================
# FOLD LOOP
# ============================================================

print("\n" + "=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)


for fold_id, (
    train_idx,
    valid_idx
) in enumerate(
    skf.split(
        np.zeros(len(y)),
        y
    ),
    start=1
):

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"FOLD {fold_id}/{N_SPLITS}"
    )

    print(
        f"Train: {len(train_idx)}"
    )

    print(
        f"Valid: {len(valid_idx)}"
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    valid_df = df.iloc[
        valid_idx
    ].copy()

    train_y = train_df[
        "label"
    ].astype(int).to_numpy()

    valid_y = valid_df[
        "label"
    ].astype(int).to_numpy()

    print(
        "Train class counts:",
        np.bincount(
            train_y
        )
    )

    print(
        "Valid class counts:",
        np.bincount(
            valid_y
        )
    )


    # ========================================================
    # FIT CALIBRATORS USING TRAINING FOLD ONLY
    # ========================================================

    percentile_model = fit_percentile(
        train_df
    )

    platt_model = fit_platt(
        train_df
    )

    isotonic_model = fit_isotonic(
        train_df
    )


    # ========================================================
    # TRANSFORM TRAINING FOLD
    # ========================================================

    train_percentile = apply_percentile(
        train_df,
        percentile_model
    )

    train_platt = apply_platt(
        train_df,
        platt_model
    )

    train_isotonic = apply_isotonic(
        train_df,
        isotonic_model
    )


    # ========================================================
    # TRANSFORM HELD-OUT FOLD
    # ========================================================

    valid_percentile = apply_percentile(
        valid_df,
        percentile_model
    )

    valid_platt = apply_platt(
        valid_df,
        platt_model
    )

    valid_isotonic = apply_isotonic(
        valid_df,
        isotonic_model
    )


    # ========================================================
    # OPTIMIZE WEIGHTS ON TRAINING FOLD ONLY
    # ========================================================

    best_p, search_p = optimize_weights(
        train_percentile,
        train_y
    )

    best_l, search_l = optimize_weights(
        train_platt,
        train_y
    )

    best_i, search_i = optimize_weights(
        train_isotonic,
        train_y
    )


    # ========================================================
    # SAVE WEIGHTS
    # ========================================================

    weight_rows.extend([

        {
            "fold":
                fold_id,

            "calibration":
                "percentile",

            "w_yolo":
                best_p["w_yolo"],

            "w_vae":
                best_p["w_vae"],

            "w_flow":
                best_p["w_flow"],

            "train_PR_AUC":
                best_p["PR_AUC"],

            "train_ROC_AUC":
                best_p["ROC_AUC"],

            "train_Best_F1":
                best_p["Best_F1"],
        },

        {
            "fold":
                fold_id,

            "calibration":
                "platt",

            "w_yolo":
                best_l["w_yolo"],

            "w_vae":
                best_l["w_vae"],

            "w_flow":
                best_l["w_flow"],

            "train_PR_AUC":
                best_l["PR_AUC"],

            "train_ROC_AUC":
                best_l["ROC_AUC"],

            "train_Best_F1":
                best_l["Best_F1"],
        },

        {
            "fold":
                fold_id,

            "calibration":
                "isotonic",

            "w_yolo":
                best_i["w_yolo"],

            "w_vae":
                best_i["w_vae"],

            "w_flow":
                best_i["w_flow"],

            "train_PR_AUC":
                best_i["PR_AUC"],

            "train_ROC_AUC":
                best_i["ROC_AUC"],

            "train_Best_F1":
                best_i["Best_F1"],
        },

    ])


    # ========================================================
    # EVALUATE HELD-OUT FOLD
    # ========================================================

    fold_calibrated = {

        "percentile":
            (
                valid_percentile,
                best_p
            ),

        "platt":
            (
                valid_platt,
                best_l
            ),

        "isotonic":
            (
                valid_isotonic,
                best_i
            ),
    }


    for method, (
        calibrated,
        weights
    ) in fold_calibrated.items():

        wy = weights[
            "w_yolo"
        ]

        wv = weights[
            "w_vae"
        ]

        wf = weights[
            "w_flow"
        ]

        scores = fuse(
            calibrated,
            wy,
            wv,
            wf
        )

        metrics = evaluate_scores(
            valid_y,
            scores
        )

        # Store OOF prediction
        oof_storage[
            method
        ][valid_idx] = scores

        fold_results.append({

            "fold":
                fold_id,

            "calibration":
                method,

            "fusion":
                "optimized",

            "w_yolo":
                wy,

            "w_vae":
                wv,

            "w_flow":
                wf,

            "valid_N":
                len(valid_idx),

            "valid_positives":
                int(
                    np.sum(
                        valid_y == 1
                    )
                ),

            "valid_negatives":
                int(
                    np.sum(
                        valid_y == 0
                    )
                ),

            "valid_ROC_AUC":
                metrics["ROC_AUC"],

            "valid_PR_AUC":
                metrics["PR_AUC"],

            "valid_Best_F1":
                metrics["Best_F1"],

            "valid_Best_F1_precision":
                metrics[
                    "Best_F1_precision"
                ],

            "valid_Best_F1_recall":
                metrics[
                    "Best_F1_recall"
                ],
        })


        # ----------------------------------------------------
        # Equal-weight result on this fold
        # ----------------------------------------------------

        equal_scores = fuse(
            calibrated,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0
        )

        equal_metrics = evaluate_scores(
            valid_y,
            equal_scores
        )

        fold_results.append({

            "fold":
                fold_id,

            "calibration":
                method,

            "fusion":
                "equal",

            "w_yolo":
                1.0 / 3.0,

            "w_vae":
                1.0 / 3.0,

            "w_flow":
                1.0 / 3.0,

            "valid_N":
                len(valid_idx),

            "valid_positives":
                int(
                    np.sum(
                        valid_y == 1
                    )
                ),

            "valid_negatives":
                int(
                    np.sum(
                        valid_y == 0
                    )
                ),

            "valid_ROC_AUC":
                equal_metrics[
                    "ROC_AUC"
                ],

            "valid_PR_AUC":
                equal_metrics[
                    "PR_AUC"
                ],

            "valid_Best_F1":
                equal_metrics[
                    "Best_F1"
                ],

            "valid_Best_F1_precision":
                equal_metrics[
                    "Best_F1_precision"
                ],

            "valid_Best_F1_recall":
                equal_metrics[
                    "Best_F1_recall"
                ],
        })


    # ========================================================
    # PRINT FOLD RESULTS
    # ========================================================

    print("\nSelected weights:")

    print(
        f"  Percentile: "
        f"{best_p['w_yolo']:.2f}, "
        f"{best_p['w_vae']:.2f}, "
        f"{best_p['w_flow']:.2f}"
    )

    print(
        f"  Platt     : "
        f"{best_l['w_yolo']:.2f}, "
        f"{best_l['w_vae']:.2f}, "
        f"{best_l['w_flow']:.2f}"
    )

    print(
        f"  Isotonic  : "
        f"{best_i['w_yolo']:.2f}, "
        f"{best_i['w_vae']:.2f}, "
        f"{best_i['w_flow']:.2f}"
    )


# ============================================================
# DATAFRAMES
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

weight_df = pd.DataFrame(
    weight_rows
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results_df.to_csv(
    OUTPUT_DIR
    / "fold_results.csv",
    index=False
)

weight_df.to_csv(
    OUTPUT_DIR
    / "weight_stability.csv",
    index=False
)


# ============================================================
# OOF EVALUATION
#
# Every validation image receives exactly one prediction from
# a model/calibration that did NOT train on that image.
# ============================================================

oof_rows = []


for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    scores = oof_storage[
        method
    ]

    if np.any(
        np.isnan(scores)
    ):

        raise RuntimeError(
            f"Missing OOF scores for {method}"
        )

    metrics = evaluate_scores(
        y,
        scores
    )

    oof_rows.append({

        "calibration":
            method,

        "OOF_ROC_AUC":
            metrics["ROC_AUC"],

        "OOF_PR_AUC":
            metrics["PR_AUC"],

        "OOF_Best_F1":
            metrics["Best_F1"],

        "OOF_Best_F1_precision":
            metrics[
                "Best_F1_precision"
            ],

        "OOF_Best_F1_recall":
            metrics[
                "Best_F1_recall"
            ],
    })


oof_results = pd.DataFrame(
    oof_rows
)

oof_results.to_csv(
    OUTPUT_DIR
    / "oof_results.csv",
    index=False
)


# ============================================================
# WEIGHT STABILITY SUMMARY
# ============================================================

weight_summary_rows = []


for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    subset = weight_df[
        weight_df["calibration"]
        == method
    ]

    for weight_name in [
        "w_yolo",
        "w_vae",
        "w_flow",
    ]:

        values = subset[
            weight_name
        ].to_numpy(
            dtype=np.float64
        )

        weight_summary_rows.append({

            "calibration":
                method,

            "weight":
                weight_name,

            "mean":
                float(
                    np.mean(values)
                ),

            "std":
                float(
                    np.std(values)
                ),

            "min":
                float(
                    np.min(values)
                ),

            "max":
                float(
                    np.max(values)
                ),

            "median":
                float(
                    np.median(values)
                ),
        })


weight_summary = pd.DataFrame(
    weight_summary_rows
)

weight_summary.to_csv(
    OUTPUT_DIR
    / "weight_summary.csv",
    index=False
)


# ============================================================
# FOLD PERFORMANCE SUMMARY
# ============================================================

performance_summary = (
    fold_results_df
    .groupby(
        [
            "calibration",
            "fusion",
        ]
    )
    .agg(
        valid_ROC_AUC_mean=(
            "valid_ROC_AUC",
            "mean"
        ),

        valid_ROC_AUC_std=(
            "valid_ROC_AUC",
            "std"
        ),

        valid_PR_AUC_mean=(
            "valid_PR_AUC",
            "mean"
        ),

        valid_PR_AUC_std=(
            "valid_PR_AUC",
            "std"
        ),

        valid_Best_F1_mean=(
            "valid_Best_F1",
            "mean"
        ),

        valid_Best_F1_std=(
            "valid_Best_F1",
            "std"
        ),
    )
    .reset_index()
)

performance_summary.to_csv(
    OUTPUT_DIR
    / "performance_summary.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FOLD PERFORMANCE SUMMARY")
print("=" * 70)

print(
    performance_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("OOF PERFORMANCE")
print("=" * 70)

print(
    oof_results.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("WEIGHT STABILITY")
print("=" * 70)

print(
    weight_summary.to_string(
        index=False
    )
)


# ============================================================
# PRINT RAW WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("WEIGHTS SELECTED IN EACH FOLD")
print("=" * 70)

print(
    weight_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE OOF PER-IMAGE SCORES
# ============================================================

oof_df = df[
    [
        c
        for c in [
            "output_image",
            "path",
            "dataset",
            "group",
            "label",
        ]
        if c in df.columns
    ]
].copy()

for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    oof_df[
        "oof_" + method
    ] = oof_storage[
        method
    ]

oof_df.to_csv(
    OUTPUT_DIR
    / "oof_per_image_scores.csv",
    index=False
)


# ============================================================
# FINAL INTERPRETATION HELPERS
# ============================================================

# Determine most stable method by mean total weight std.

stability_scores = {}

for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    subset = weight_summary[
        weight_summary["calibration"]
        == method
    ]

    total_std = float(
        subset["std"].sum()
    )

    stability_scores[
        method
    ] = total_std


most_stable = min(
    stability_scores,
    key=stability_scores.get
)


# Highest OOF PR-AUC

best_oof_idx = int(
    oof_results[
        "OOF_PR_AUC"
    ].idxmax()
)

best_oof_method = (
    oof_results.loc[
        best_oof_idx,
        "calibration"
    ]
)

best_oof_pr = float(
    oof_results.loc[
        best_oof_idx,
        "OOF_PR_AUC"
    ]
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "calibration_cv_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD — 5-FOLD CALIBRATION / FUSION STABILITY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "IMPORTANT METHODOLOGY\n"
    )

    f.write(
        "Only the validation set was used.\n"
    )

    f.write(
        "The test set was not accessed.\n"
    )

    f.write(
        "Each fold's calibrator was fitted only on its\n"
    )

    f.write(
        "training portion, then evaluated on its held-out fold.\n\n"
    )

    f.write(
        "OOF RESULTS\n"
    )

    f.write(
        oof_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nPER-FOLD PERFORMANCE\n"
    )

    f.write(
        performance_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nWEIGHT STABILITY\n"
    )

    f.write(
        weight_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nMOST STABLE CALIBRATION BY TOTAL WEIGHT STD\n"
    )

    f.write(
        f"{most_stable}\n"
    )

    f.write(
        "\nBEST OOF PR-AUC METHOD\n"
    )

    f.write(
        f"{best_oof_method}\n"
    )

    f.write(
        f"OOF PR-AUC = {best_oof_pr:.6f}\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nImportant files:"
)

print(
    "  fold_results.csv"
)

print(
    "  weight_stability.csv"
)

print(
    "  weight_summary.csv"
)

print(
    "  performance_summary.csv"
)

print(
    "  oof_results.csv"
)

print(
    "  oof_per_image_scores.csv"
)

print(
    "  calibration_cv_summary.txt"
)