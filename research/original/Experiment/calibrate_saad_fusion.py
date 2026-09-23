# ============================================================
# SAAD — SCORE CALIBRATION EXPERIMENT
# ============================================================
#
# Compare:
#
#   A. Percentile normalization
#   B. Platt / logistic calibration
#   C. Isotonic calibration
#
# Then compare:
#
#   - equal-weight fusion
#   - validation-optimized fusion
#
# IMPORTANT:
#   * Models are NOT retrained.
#   * Calibration is fitted ONLY on validation data.
#   * Fusion weights are selected ONLY on validation data.
#   * Test labels are used ONLY for final evaluation.
#
# ============================================================

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42

RESULT_ROOT = Path(
    r"E:\SIH\SIH_Results"
)

INPUT_DIR = (
    RESULT_ROOT
    / "validation_weighted_fusion"
)

OUTPUT_DIR = (
    RESULT_ROOT
    / "score_calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_FILE = (
    INPUT_DIR
    / "validation_raw_scores.csv"
)

TEST_FILE = (
    INPUT_DIR
    / "test_raw_scores.csv"
)

WEIGHT_STEP = 0.05


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CHECK INPUTS
# ============================================================

if not VAL_FILE.exists():

    raise FileNotFoundError(
        f"Validation score file not found:\n{VAL_FILE}"
    )

if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test score file not found:\n{TEST_FILE}"
    )


# ============================================================
# LOAD SCORES
# ============================================================

print("=" * 70)
print("SAAD — SCORE CALIBRATION EXPERIMENT")
print("=" * 70)

print("\nLoading cached scores...")

val = pd.read_csv(
    VAL_FILE
)

test = pd.read_csv(
    TEST_FILE
)

print(
    "Validation images:",
    len(val)
)

print(
    "Test images      :",
    len(test)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "label",
    "yolo_raw",
    "vae_raw",
    "flow_raw",
]

for col in required:

    if col not in val.columns:

        raise RuntimeError(
            f"Missing validation column: {col}"
        )

    if col not in test.columns:

        raise RuntimeError(
            f"Missing test column: {col}"
        )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

val = val.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=required
).reset_index(
    drop=True
)

test = test.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=required
).reset_index(
    drop=True
)


# ============================================================
# LABEL ARRAYS
# ============================================================

y_val = val[
    "label"
].astype(int).to_numpy()

y_test = test[
    "label"
].astype(int).to_numpy()


print("\nValidation label distribution:")

print(
    pd.Series(
        y_val
    ).value_counts().sort_index()
)

print("\nTest label distribution:")

print(
    pd.Series(
        y_test
    ).value_counts().sort_index()
)


# ============================================================
# SCORE NAMES
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
# 1. PERCENTILE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("FITTING PERCENTILE NORMALIZATION")
print("=" * 70)

percentile_stats = {}

# IMPORTANT:
# Use NORMAL validation data to establish score scale.
#
# We do NOT use anthropogenic validation images to define
# the normal score distribution.

normal_val = val[
    val["label"] == 0
].copy()

if len(normal_val) < 20:

    raise RuntimeError(
        "Too few normal validation images."
    )


for signal in SIGNALS:

    raw_col = RAW_COLUMNS[
        signal
    ]

    p5 = float(
        np.percentile(
            normal_val[
                raw_col
            ],
            5
        )
    )

    p95 = float(
        np.percentile(
            normal_val[
                raw_col
            ],
            95
        )
    )

    if p95 <= p5:

        p95 = p5 + 1e-8

    percentile_stats[
        signal
    ] = {
        "p5": p5,
        "p95": p95,
    }

    print(
        f"{signal:6s} "
        f"P5={p5:.8f} "
        f"P95={p95:.8f}"
    )


def percentile_transform(
    values,
    p5,
    p95
):

    z = (
        np.asarray(values)
        - p5
    ) / (
        p95 - p5
    )

    return np.clip(
        z,
        0.0,
        1.0
    )


# ============================================================
# 2. PLATT CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("FITTING PLATT CALIBRATION")
print("=" * 70)

platt_models = {}


for signal in SIGNALS:

    raw_col = RAW_COLUMNS[
        signal
    ]

    x = val[
        [raw_col]
    ].to_numpy(
        dtype=np.float64
    )

    # Logistic regression maps the raw anomaly score
    # to a probability-like calibrated score.
    #
    # class_weight balances normal/anthropogenic validation
    # samples if the split is imbalanced.

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    model.fit(
        x,
        y_val
    )

    platt_models[
        signal
    ] = model

    print(
        f"{signal:6s} fitted"
    )


def apply_platt(
    df,
    signal
):

    raw_col = RAW_COLUMNS[
        signal
    ]

    x = df[
        [raw_col]
    ].to_numpy(
        dtype=np.float64
    )

    return platt_models[
        signal
    ].predict_proba(
        x
    )[:, 1]


# ============================================================
# 3. ISOTONIC CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("FITTING ISOTONIC CALIBRATION")
print("=" * 70)

isotonic_models = {}


for signal in SIGNALS:

    raw_col = RAW_COLUMNS[
        signal
    ]

    x = val[
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
        y_val
    )

    isotonic_models[
        signal
    ] = model

    print(
        f"{signal:6s} fitted"
    )


def apply_isotonic(
    df,
    signal
):

    raw_col = RAW_COLUMNS[
        signal
    ]

    x = df[
        raw_col
    ].to_numpy(
        dtype=np.float64
    )

    return isotonic_models[
        signal
    ].predict(
        x
    )


# ============================================================
# GENERATE CALIBRATED SCORES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CALIBRATED SCORES")
print("=" * 70)


# ------------------------------------------------------------
# Percentile
# ------------------------------------------------------------

for signal in SIGNALS:

    raw_col = RAW_COLUMNS[
        signal
    ]

    stats = percentile_stats[
        signal
    ]

    val[
        signal + "_percentile"
    ] = percentile_transform(
        val[raw_col],
        stats["p5"],
        stats["p95"]
    )

    test[
        signal + "_percentile"
    ] = percentile_transform(
        test[raw_col],
        stats["p5"],
        stats["p95"]
    )


# ------------------------------------------------------------
# Platt
# ------------------------------------------------------------

for signal in SIGNALS:

    val[
        signal + "_platt"
    ] = apply_platt(
        val,
        signal
    )

    test[
        signal + "_platt"
    ] = apply_platt(
        test,
        signal
    )


# ------------------------------------------------------------
# Isotonic
# ------------------------------------------------------------

for signal in SIGNALS:

    val[
        signal + "_isotonic"
    ] = apply_isotonic(
        val,
        signal
    )

    test[
        signal + "_isotonic"
    ] = apply_isotonic(
        test,
        signal
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    y,
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    roc = roc_auc_score(
        y,
        scores
    )

    pr = average_precision_score(
        y,
        scores
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            scores
        )
    )

    f1_values = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    idx = int(
        np.argmax(
            f1_values
        )
    )

    best_f1 = float(
        f1_values[idx]
    )

    best_precision = float(
        precision[idx]
    )

    best_recall = float(
        recall[idx]
    )

    if idx < len(thresholds):

        best_threshold = float(
            thresholds[idx]
        )

    else:

        best_threshold = float(
            np.median(scores)
        )

    return {
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "Best_F1": best_f1,
        "Best_F1_precision":
            best_precision,
        "Best_F1_recall":
            best_recall,
        "Best_F1_threshold":
            best_threshold,
    }


# ============================================================
# INDIVIDUAL SIGNAL CALIBRATION RESULTS
# ============================================================

individual_rows = []


for signal in SIGNALS:

    for method in [
        "percentile",
        "platt",
        "isotonic",
    ]:

        column = (
            signal
            + "_"
            + method
        )

        val_metrics = evaluate(
            y_val,
            val[column]
        )

        test_metrics = evaluate(
            y_test,
            test[column]
        )

        individual_rows.append({

            "signal":
                signal,

            "calibration":
                method,

            "val_ROC_AUC":
                val_metrics["ROC_AUC"],

            "val_PR_AUC":
                val_metrics["PR_AUC"],

            "val_Best_F1":
                val_metrics["Best_F1"],

            "test_ROC_AUC":
                test_metrics["ROC_AUC"],

            "test_PR_AUC":
                test_metrics["PR_AUC"],

            "test_Best_F1":
                test_metrics["Best_F1"],
        })


individual_results = pd.DataFrame(
    individual_rows
)

individual_results.to_csv(
    OUTPUT_DIR
    / "individual_calibration_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("INDIVIDUAL SIGNAL RESULTS")
print("=" * 70)

print(
    individual_results.to_string(
        index=False
    )
)


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
                    round(wy, 6),
                    round(wv, 6),
                    round(wf, 6),
                )
            )

    return weights


weight_grid = generate_weights(
    WEIGHT_STEP
)


# ============================================================
# FUSION SEARCH
# ============================================================

def search_fusion(
    val_df,
    calibration_method
):

    y = val_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    yolo = val_df[
        "yolo_" + calibration_method
    ].to_numpy(
        dtype=np.float64
    )

    vae = val_df[
        "vae_" + calibration_method
    ].to_numpy(
        dtype=np.float64
    )

    flow = val_df[
        "flow_" + calibration_method
    ].to_numpy(
        dtype=np.float64
    )

    results = []

    for wy, wv, wf in weight_grid:

        fused = (
            wy * yolo
            + wv * vae
            + wf * flow
        )

        roc = roc_auc_score(
            y,
            fused
        )

        pr = average_precision_score(
            y,
            fused
        )

        precision, recall, thresholds = (
            precision_recall_curve(
                y,
                fused
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

        results.append({

            "calibration":
                calibration_method,

            "w_yolo":
                wy,

            "w_vae":
                wv,

            "w_flow":
                wf,

            "val_ROC_AUC":
                roc,

            "val_PR_AUC":
                pr,

            "val_Best_F1":
                float(
                    f1[best_idx]
                ),
        })

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        [
            "val_PR_AUC",
            "val_ROC_AUC",
            "val_Best_F1",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# SEARCH ALL CALIBRATION METHODS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FUSION SEARCH")
print("=" * 70)

all_weight_results = []

best_weights = {}


for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    print(
        f"\nSearching: {method}"
    )

    result = search_fusion(
        val,
        method
    )

    all_weight_results.append(
        result
    )

    best_row = result.iloc[0]

    best_weights[
        method
    ] = (
        float(best_row["w_yolo"]),
        float(best_row["w_vae"]),
        float(best_row["w_flow"]),
    )

    print(
        f"Best weights: "
        f"YOLO={best_row['w_yolo']:.2f}, "
        f"VAE={best_row['w_vae']:.2f}, "
        f"Flow={best_row['w_flow']:.2f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_row['val_PR_AUC']:.6f}"
    )


all_weight_results_df = pd.concat(
    all_weight_results,
    ignore_index=True
)

all_weight_results_df.to_csv(
    OUTPUT_DIR
    / "all_weight_search_results.csv",
    index=False
)


# ============================================================
# TEST EVALUATION OF VALIDATION-SELECTED WEIGHTS
# ============================================================

fusion_rows = []


for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    wy, wv, wf = best_weights[
        method
    ]

    column_yolo = (
        "yolo_" + method
    )

    column_vae = (
        "vae_" + method
    )

    column_flow = (
        "flow_" + method
    )

    val_fused = (
        wy * val[column_yolo]
        + wv * val[column_vae]
        + wf * val[column_flow]
    )

    test_fused = (
        wy * test[column_yolo]
        + wv * test[column_vae]
        + wf * test[column_flow]
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    val_metrics = evaluate(
        y_val,
        val_fused
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_metrics = evaluate(
        y_test,
        test_fused
    )

    fusion_rows.append({

        "calibration":
            method,

        "fusion":
            "validation_optimized",

        "w_yolo":
            wy,

        "w_vae":
            wv,

        "w_flow":
            wf,

        "val_ROC_AUC":
            val_metrics["ROC_AUC"],

        "val_PR_AUC":
            val_metrics["PR_AUC"],

        "val_Best_F1":
            val_metrics["Best_F1"],

        "test_ROC_AUC":
            test_metrics["ROC_AUC"],

        "test_PR_AUC":
            test_metrics["PR_AUC"],

        "test_Best_F1":
            test_metrics["Best_F1"],
    })


    # --------------------------------------------------------
    # Equal weights
    # --------------------------------------------------------

    val_equal = (
        val[column_yolo]
        + val[column_vae]
        + val[column_flow]
    ) / 3.0

    test_equal = (
        test[column_yolo]
        + test[column_vae]
        + test[column_flow]
    ) / 3.0

    val_equal_metrics = evaluate(
        y_val,
        val_equal
    )

    test_equal_metrics = evaluate(
        y_test,
        test_equal
    )

    fusion_rows.append({

        "calibration":
            method,

        "fusion":
            "equal_weight",

        "w_yolo":
            1.0 / 3.0,

        "w_vae":
            1.0 / 3.0,

        "w_flow":
            1.0 / 3.0,

        "val_ROC_AUC":
            val_equal_metrics["ROC_AUC"],

        "val_PR_AUC":
            val_equal_metrics["PR_AUC"],

        "val_Best_F1":
            val_equal_metrics["Best_F1"],

        "test_ROC_AUC":
            test_equal_metrics["ROC_AUC"],

        "test_PR_AUC":
            test_equal_metrics["PR_AUC"],

        "test_Best_F1":
            test_equal_metrics["Best_F1"],
    })


fusion_results = pd.DataFrame(
    fusion_rows
)

fusion_results = fusion_results.sort_values(
    [
        "test_PR_AUC",
        "test_ROC_AUC",
    ],
    ascending=False
).reset_index(
    drop=True
)

fusion_results.to_csv(
    OUTPUT_DIR
    / "calibrated_fusion_results.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION + FUSION RESULTS")
print("=" * 70)

print(
    fusion_results.to_string(
        index=False
    )
)


# ============================================================
# SELECTED TEST SCORE COLUMNS
# ============================================================

for method in [
    "percentile",
    "platt",
    "isotonic",
]:

    wy, wv, wf = best_weights[
        method
    ]

    test[
        "fusion_"
        + method
        + "_weighted"
    ] = (
        wy * test[
            "yolo_" + method
        ]
        + wv * test[
            "vae_" + method
        ]
        + wf * test[
            "flow_" + method
        ]
    )

    test[
        "fusion_"
        + method
        + "_equal"
    ] = (
        test[
            "yolo_" + method
        ]
        + test[
            "vae_" + method
        ]
        + test[
            "flow_" + method
        ]
    ) / 3.0


# ============================================================
# SAVE PER-IMAGE RESULTS
# ============================================================

test.to_csv(
    OUTPUT_DIR
    / "per_image_calibrated_scores.csv",
    index=False
)


# ============================================================
# DOMAIN-WISE RESULTS
# ============================================================

domain_rows = []


if "dataset" in test.columns:

    for dataset_name, group in test.groupby(
        "dataset"
    ):

        y = group[
            "label"
        ].to_numpy(
            dtype=np.int32
        )

        for method in [
            "percentile",
            "platt",
            "isotonic",
        ]:

            score_col = (
                "fusion_"
                + method
                + "_weighted"
            )

            scores = group[
                score_col
            ].to_numpy(
                dtype=np.float64
            )

            if len(
                np.unique(y)
            ) < 2:

                roc = np.nan
                pr = np.nan

            else:

                roc = roc_auc_score(
                    y,
                    scores
                )

                pr = average_precision_score(
                    y,
                    scores
                )

            domain_rows.append({

                "dataset":
                    dataset_name,

                "calibration":
                    method,

                "N":
                    len(group),

                "positives":
                    int(
                        np.sum(y == 1)
                    ),

                "negatives":
                    int(
                        np.sum(y == 0)
                    ),

                "ROC_AUC":
                    roc,

                "PR_AUC":
                    pr,
            })


domain_results = pd.DataFrame(
    domain_rows
)

domain_results.to_csv(
    OUTPUT_DIR
    / "dataset_wise_calibration_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("DATASET-WISE RESULTS")
print("=" * 70)

if len(domain_results) > 0:

    print(
        domain_results.to_string(
            index=False
        )
    )


# ============================================================
# SAVE NORMALIZATION / MODEL INFORMATION
# ============================================================

with open(
    OUTPUT_DIR
    / "calibration_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD — SCORE CALIBRATION EXPERIMENT\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "METHODOLOGY\n"
    )

    f.write(
        "Calibration fitted using validation data only.\n"
    )

    f.write(
        "Fusion weights selected using validation data only.\n"
    )

    f.write(
        "Test data was used only for final evaluation.\n\n"
    )

    f.write(
        "PERCENTILE NORMALIZATION\n"
    )

    for signal, stats in (
        percentile_stats.items()
    ):

        f.write(
            f"{signal}: "
            f"P5={stats['p5']:.10f}, "
            f"P95={stats['p95']:.10f}\n"
        )

    f.write(
        "\nBEST VALIDATION WEIGHTS\n"
    )

    for method, weights in (
        best_weights.items()
    ):

        wy, wv, wf = weights

        f.write(
            f"{method}: "
            f"YOLO={wy:.4f}, "
            f"VAE={wv:.4f}, "
            f"Flow={wf:.4f}\n"
        )

    f.write(
        "\nRESULTS\n"
    )

    f.write(
        fusion_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nFILES\n"
    )

    f.write(
        "individual_calibration_results.csv\n"
    )

    f.write(
        "all_weight_search_results.csv\n"
    )

    f.write(
        "calibrated_fusion_results.csv\n"
    )

    f.write(
        "per_image_calibrated_scores.csv\n"
    )

    f.write(
        "dataset_wise_calibration_results.csv\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "\nCalibration methods tested:"
)

print(
    "  1. Percentile normalization"
)

print(
    "  2. Platt / logistic calibration"
)

print(
    "  3. Isotonic calibration"
)

print(
    "\nValidation-selected weights:"
)

for method, weights in (
    best_weights.items()
):

    wy, wv, wf = weights

    print(
        f"  {method:10s}: "
        f"YOLO={wy:.2f}, "
        f"VAE={wv:.2f}, "
        f"Flow={wf:.2f}"
    )


print(
    "\nFinal test comparison:"
)

print(
    fusion_results[
        [
            "calibration",
            "fusion",
            "w_yolo",
            "w_vae",
            "w_flow",
            "test_ROC_AUC",
            "test_PR_AUC",
            "test_Best_F1",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDONE."
)