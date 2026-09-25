"""
SAAD FINAL UNTOUCHED TEST EVALUATION
====================================

Purpose
-------
Final evaluation of the frozen SAAD evidence-ranking system.

IMPORTANT:
- Test labels are NEVER used for normalization or threshold selection.
- Normalization is fitted ONLY on NORMAL validation images.
- Thresholds are fitted ONLY on NORMAL validation images.
- Test labels are used ONLY after all decisions are frozen, for reporting metrics.
- No model training occurs here.
- No weights are modified.

Primary evidence policies:
    YOLO
    VAE
    FLOW
    MAX
    MEAN
    TOP2_MEAN
    YOLO_FLOW_MEAN
    YOLO_VAE_MEAN
    VAE_FLOW_MEAN

Primary recommended policy:
    YOLO_FLOW_MEAN

Outputs:
    E:\\SIH\\SIH_Results\\final_saad_evaluation\\
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"E:\SIH\Datasets\SAAD_baseline")

RESULT_ROOT = Path(r"E:\SIH\SIH_Results")

RAW_ROOT = RESULT_ROOT / "validation_weighted_fusion"

VAL_RAW_CSV = RAW_ROOT / "validation_raw_scores.csv"
TEST_RAW_CSV = RAW_ROOT / "test_raw_scores.csv"

OUT_ROOT = RESULT_ROOT / "final_saad_evaluation"

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# Validation-normal percentile normalization.
# These are FIT from validation normals inside this script.
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 95.0

# Thresholds are derived from NORMAL VALIDATION ONLY.
FIXED_NORMAL_PERCENTILES = [95.0, 99.0]

PRIMARY_POLICY = "YOLO_FLOW_MEAN"


# ============================================================
# UTILITIES
# ============================================================

def clean_name(x):
    return str(x).strip().lower()


def find_column(df, candidates, required=True):
    """
    Find a dataframe column using case-insensitive matching.
    """
    normalized = {clean_name(c): c for c in df.columns}

    for candidate in candidates:
        key = clean_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not find any of {candidates}.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def ensure_binary_label(df):
    """
    Locate / normalize the image-level binary label.

    Expected semantics:
        0 = normal
        1 = anthropogenic
    """

    col = find_column(
        df,
        [
            "label",
            "target",
            "y",
            "class",
            "is_anthropogenic",
            "anthropogenic",
            "has_annotation",
        ],
        required=False,
    )

    if col is None:
        raise KeyError(
            "Could not identify the binary image-level label."
        )

    vals = df[col]

    if pd.api.types.is_numeric_dtype(vals):
        out = pd.to_numeric(vals, errors="coerce")
    else:
        mapping = {
            "normal": 0,
            "background": 0,
            "negative": 0,
            "0": 0,
            "false": 0,
            "anthropogenic": 1,
            "positive": 1,
            "1": 1,
            "true": 1,
        }

        out = vals.astype(str).str.strip().str.lower().map(mapping)

    if out.isna().any():
        bad = df.loc[out.isna(), col].astype(str).unique()
        raise ValueError(
            f"Unrecognized label values in column '{col}': {bad[:20]}"
        )

    out = out.astype(int)

    if not set(out.unique()).issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0/1 but found: {sorted(out.unique())}"
        )

    return out


def ensure_dataset_column(df):
    col = find_column(
        df,
        [
            "dataset",
            "source_dataset",
            "source",
            "domain",
        ],
        required=False,
    )

    if col is None:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)

    return df[col].astype(str)


def load_raw_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired raw-score file does not exist:\n{path}\n"
            f"\nExpected cached files from the previous fusion evaluation."
        )

    df = pd.read_csv(path)

    print("=" * 70)
    print(f"LOADED: {path}")
    print("=" * 70)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {list(df.columns)}")

    return df


# ============================================================
# RAW SCORE COLUMN DETECTION
# ============================================================

def detect_score_columns(df):
    """
    Locate the raw image-level evidence scores.

    Required:
        YOLO
        VAE
        FLOW
    """

    yolo = find_column(
        df,
        [
            "yolo_raw",
            "yolo_score",
            "yolo_conf",
            "yolo",
        ],
    )

    vae = find_column(
        df,
        [
            "vae_raw",
            "vae_score",
            "vae_anomaly",
            "vae",
        ],
    )

    flow = find_column(
        df,
        [
            "flow_raw",
            "flow_score",
            "flow_nll",
            "flow_anomaly",
            "flow",
        ],
    )

    return yolo, vae, flow


# ============================================================
# NORMALIZATION
# ============================================================

def fit_normalization(val_df):
    """
    Fit P5/P95 normalization using ONLY validation NORMAL images.
    """

    yolo_col, vae_col, flow_col = detect_score_columns(val_df)

    normal = val_df[val_df["label"] == 0].copy()

    if len(normal) == 0:
        raise RuntimeError(
            "Validation set contains no normal images."
        )

    params = {}

    print("\n" + "=" * 70)
    print("FITTING NORMALIZATION ON VALIDATION NORMALS ONLY")
    print("=" * 70)
    print(f"Validation total : {len(val_df)}")
    print(f"Validation normal: {len(normal)}")

    for name, col in [
        ("YOLO", yolo_col),
        ("VAE", vae_col),
        ("FLOW", flow_col),
    ]:

        x = pd.to_numeric(
            normal[col],
            errors="coerce"
        ).values

        x = x[np.isfinite(x)]

        lo = np.percentile(x, LOW_PERCENTILE)
        hi = np.percentile(x, HIGH_PERCENTILE)

        if hi <= lo:
            raise RuntimeError(
                f"Invalid normalization range for {name}: "
                f"{lo} -> {hi}"
            )

        params[name] = {
            "column": col,
            "p5": float(lo),
            "p95": float(hi),
        }

        print(
            f"{name:5s} | "
            f"N={len(x):4d} | "
            f"P5={lo:.8f} | "
            f"P95={hi:.8f}"
        )

    return params


def apply_normalization(df, params):
    """
    Percentile-normalize each evidence signal.

    Values outside [P5,P95] are clipped.

    IMPORTANT:
    params come from validation normals only.
    """

    out = df.copy()

    for name in ["YOLO", "VAE", "FLOW"]:

        col = params[name]["column"]
        lo = params[name]["p5"]
        hi = params[name]["p95"]

        raw = pd.to_numeric(
            out[col],
            errors="coerce"
        ).values

        norm = (raw - lo) / (hi - lo)

        norm = np.clip(norm, 0.0, 1.0)

        out[f"{name}_EVIDENCE"] = norm

    return out


# ============================================================
# EVIDENCE POLICIES
# ============================================================

def build_policies(df):
    """
    Build frozen evidence-ranking policies.

    No learned parameters here.
    """

    y = df["YOLO_EVIDENCE"].values
    v = df["VAE_EVIDENCE"].values
    f = df["FLOW_EVIDENCE"].values

    signals = np.column_stack([y, v, f])

    out = df.copy()

    out["YOLO"] = y
    out["VAE"] = v
    out["FLOW"] = f

    out["MAX"] = np.max(signals, axis=1)

    out["MEAN"] = np.mean(signals, axis=1)

    sorted_signals = np.sort(
        signals,
        axis=1
    )

    # Highest two signals
    out["TOP2_MEAN"] = np.mean(
        sorted_signals[:, 1:],
        axis=1
    )

    out["YOLO_FLOW_MEAN"] = (
        y + f
    ) / 2.0

    out["YOLO_VAE_MEAN"] = (
        y + v
    ) / 2.0

    out["VAE_FLOW_MEAN"] = (
        v + f
    ) / 2.0

    return out


POLICIES = [
    "YOLO",
    "VAE",
    "FLOW",
    "MAX",
    "MEAN",
    "TOP2_MEAN",
    "YOLO_FLOW_MEAN",
    "YOLO_VAE_MEAN",
    "VAE_FLOW_MEAN",
]


# ============================================================
# METRICS
# ============================================================

def safe_auc(y_true, score):
    """
    ROC-AUC requires both classes.
    """

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score
        )
    )


def safe_pr_auc(y_true, score):
    """
    Average precision also requires meaningful class structure.
    """

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            score
        )
    )


def best_f1_diagnostic(y_true, score):
    """
    TEST-SET ORACLE DIAGNOSTIC ONLY.

    This MUST NOT be used for deployment.

    It answers:
        "If we were allowed to tune a threshold on test,
         what is the maximum F1?"

    Useful for ranking quality comparison only.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        score
    )

    if len(thresholds) == 0:
        return {
            "threshold": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    idx = int(np.nanargmax(f1))

    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def threshold_metrics(y_true, score, threshold):
    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def ranking_metrics(df, split_name):
    rows = []

    y_true = df["label"].values

    for policy in POLICIES:

        score = df[policy].values

        roc = safe_auc(
            y_true,
            score
        )

        pr = safe_pr_auc(
            y_true,
            score
        )

        best = best_f1_diagnostic(
            y_true,
            score
        )

        rows.append({
            "split": split_name,
            "policy": policy,
            "N": len(df),
            "normal": int((y_true == 0).sum()),
            "anthropogenic": int((y_true == 1).sum()),
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "TEST_ORACLE_BEST_F1": best["f1"],
            "TEST_ORACLE_THRESHOLD": best["threshold"],
            "TEST_ORACLE_PRECISION": best["precision"],
            "TEST_ORACLE_RECALL": best["recall"],
        })

    return pd.DataFrame(rows)


# ============================================================
# FIXED THRESHOLDS
# ============================================================

def fit_normal_thresholds(val_df):
    """
    Derive thresholds from VALIDATION NORMALS ONLY.

    P95:
        approximately 5% validation-normal exceedance.

    P99:
        approximately 1% validation-normal exceedance.

    These thresholds are then frozen and applied to test.
    """

    normal = val_df[
        val_df["label"] == 0
    ]

    rows = []

    for policy in POLICIES:

        scores = normal[policy].values

        for percentile in FIXED_NORMAL_PERCENTILES:

            threshold = np.percentile(
                scores,
                percentile
            )

            rows.append({
                "policy": policy,
                "normal_validation_percentile": percentile,
                "threshold": float(threshold),
                "validation_normal_N": len(scores),
            })

    return pd.DataFrame(rows)


def evaluate_fixed_thresholds(test_df, thresholds):
    rows = []

    y_true = test_df["label"].values

    for _, row in thresholds.iterrows():

        policy = row["policy"]
        threshold = row["threshold"]

        score = test_df[policy].values

        metrics = threshold_metrics(
            y_true,
            score,
            threshold
        )

        metrics.update({
            "policy": policy,
            "validation_normal_percentile":
                row["normal_validation_percentile"],
            "validation_normal_N":
                row["validation_normal_N"],
        })

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# DATASET-WISE EVALUATION
# ============================================================

def dataset_metrics(test_df):
    rows = []

    for dataset, group in test_df.groupby("dataset"):

        y_true = group["label"].values

        for policy in POLICIES:

            score = group[policy].values

            best = best_f1_diagnostic(
                y_true,
                score
            )

            rows.append({
                "dataset": dataset,
                "policy": policy,
                "N": len(group),
                "normal": int((y_true == 0).sum()),
                "anthropogenic": int((y_true == 1).sum()),
                "ROC_AUC": safe_auc(y_true, score),
                "PR_AUC": safe_pr_auc(y_true, score),
                "oracle_best_F1": best["f1"],
                "oracle_threshold": best["threshold"],
            })

    return pd.DataFrame(rows)


# ============================================================
# SAVE ROC / PR CURVES
# ============================================================

def save_curves(test_df):

    y_true = test_df["label"].values

    # ------------------------------
    # ROC
    # ------------------------------

    plt.figure(figsize=(9, 7))

    for policy in POLICIES:

        score = test_df[policy].values

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_true,
            score
        )

        auc = roc_auc_score(
            y_true,
            score
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{policy}  AUC={auc:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SAAD Final Test ROC Curves")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_ROOT / "final_test_ROC.png",
        dpi=200
    )

    plt.close()

    # ------------------------------
    # PR
    # ------------------------------

    plt.figure(figsize=(9, 7))

    for policy in POLICIES:

        score = test_df[policy].values

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(
            y_true,
            score
        )

        ap = average_precision_score(
            y_true,
            score
        )

        plt.plot(
            recall,
            precision,
            label=f"{policy}  AP={ap:.3f}"
        )

    prevalence = np.mean(y_true)

    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Random baseline={prevalence:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("SAAD Final Test Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_ROOT / "final_test_PR.png",
        dpi=200
    )

    plt.close()


# ============================================================
# FINAL VERDICT
# ============================================================

def make_verdict(
    test_metrics,
    fixed_metrics,
    test_df,
    normalization
):

    primary = test_metrics[
        test_metrics["policy"] == PRIMARY_POLICY
    ].iloc[0]

    fixed_primary = fixed_metrics[
        fixed_metrics["policy"] == PRIMARY_POLICY
    ].copy()

    lines = []

    lines.append(
        "SAAD FINAL UNTOUCHED TEST EVALUATION"
    )
    lines.append(
        "=" * 60
    )
    lines.append("")

    lines.append(
        "TEST SET"
    )
    lines.append(
        f"Images              : {len(test_df)}"
    )
    lines.append(
        f"Normal              : {(test_df['label'] == 0).sum()}"
    )
    lines.append(
        f"Anthropogenic       : {(test_df['label'] == 1).sum()}"
    )
    lines.append("")

    lines.append(
        "PRIMARY POLICY"
    )
    lines.append(
        f"Policy               : {PRIMARY_POLICY}"
    )
    lines.append(
        f"ROC-AUC              : {primary['ROC_AUC']:.6f}"
    )
    lines.append(
        f"PR-AUC               : {primary['PR_AUC']:.6f}"
    )
    lines.append(
        f"Test-oracle best F1  : {primary['TEST_ORACLE_BEST_F1']:.6f}"
    )
    lines.append("")

    lines.append(
        "IMPORTANT:"
    )
    lines.append(
        "The test-oracle F1 is a diagnostic upper-bound-style "
        "threshold result and MUST NOT be presented as the deployed "
        "operating point."
    )
    lines.append("")

    lines.append(
        "FIXED VALIDATION-NORMAL THRESHOLDS"
    )

    for _, row in fixed_primary.iterrows():

        lines.append(
            f"P{int(row['validation_normal_percentile']):02d} "
            f"threshold={row['threshold']:.6f} | "
            f"P={row['precision']:.6f} | "
            f"R={row['recall']:.6f} | "
            f"F1={row['f1']:.6f} | "
            f"TN={row['TN']} FP={row['FP']} "
            f"FN={row['FN']} TP={row['TP']}"
        )

    lines.append("")
    lines.append(
        "NORMALIZATION WAS FIT USING VALIDATION NORMALS ONLY:"
    )

    for name, p in normalization.items():

        lines.append(
            f"{name}: "
            f"P5={p['p5']:.8f}, "
            f"P95={p['p95']:.8f}"
        )

    lines.append("")
    lines.append(
        "No test labels were used to fit normalization or fixed thresholds."
    )

    lines.append("")
    lines.append(
        "RECOMMENDATION"
    )

    lines.append(
        "Use the primary policy as the main SAAD ranking signal, "
        "while presenting fixed-threshold operating points separately "
        "from test-oracle F1."
    )

    lines.append("")
    lines.append(
        "Domain-wise results must be interpreted carefully because "
        "SubPipeMini2 contains only anthropogenic test examples."
    )

    text = "\n".join(lines)

    with open(
        OUT_ROOT / "FINAL_VERDICT.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    print("\n")
    print("=" * 70)
    print("SAAD FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 70)
    print("")

    # --------------------------------------------------------
    # Load raw scores
    # --------------------------------------------------------

    val_df = load_raw_csv(
        VAL_RAW_CSV
    )

    test_df = load_raw_csv(
        TEST_RAW_CSV
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    val_df["label"] = ensure_binary_label(
        val_df
    )

    test_df["label"] = ensure_binary_label(
        test_df
    )

    # --------------------------------------------------------
    # Dataset/domain
    # --------------------------------------------------------

    val_df["dataset"] = ensure_dataset_column(
        val_df
    )

    test_df["dataset"] = ensure_dataset_column(
        test_df
    )

    # --------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SANITY CHECK")
    print("=" * 70)

    print("\nVALIDATION:")
    print(
        val_df["label"]
        .value_counts()
        .sort_index()
        .rename({0: "normal", 1: "anthropogenic"})
    )

    print("\nTEST:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
        .rename({0: "normal", 1: "anthropogenic"})
    )

    print("\nTEST DOMAINS:")
    print(
        test_df["dataset"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Fit normalization
    # --------------------------------------------------------

    normalization = fit_normalization(
        val_df
    )

    # --------------------------------------------------------
    # Apply normalization
    # --------------------------------------------------------

    val_norm = apply_normalization(
        val_df,
        normalization
    )

    test_norm = apply_normalization(
        test_df,
        normalization
    )

    # --------------------------------------------------------
    # Build frozen evidence policies
    # --------------------------------------------------------

    val_scores = build_policies(
        val_norm
    )

    test_scores = build_policies(
        test_norm
    )

    # --------------------------------------------------------
    # Save raw final test evidence
    # --------------------------------------------------------

    test_scores.to_csv(
        OUT_ROOT / "final_test_scores.csv",
        index=False
    )

    val_scores.to_csv(
        OUT_ROOT / "validation_frozen_scores.csv",
        index=False
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST RANKING PERFORMANCE")
    print("=" * 70)

    test_metrics = ranking_metrics(
        test_scores,
        "test"
    )

    test_metrics = test_metrics.sort_values(
        "PR_AUC",
        ascending=False
    )

    print(
        test_metrics[
            [
                "policy",
                "ROC_AUC",
                "PR_AUC",
                "TEST_ORACLE_BEST_F1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    test_metrics.to_csv(
        OUT_ROOT / "final_test_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Fixed thresholds from validation normals
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FITTING FIXED THRESHOLDS FROM VALIDATION NORMALS")
    print("=" * 70)

    thresholds = fit_normal_thresholds(
        val_scores
    )

    thresholds.to_csv(
        OUT_ROOT / "validation_normal_thresholds.csv",
        index=False
    )

    # --------------------------------------------------------
    # Apply thresholds to untouched test
    # --------------------------------------------------------

    fixed_metrics = evaluate_fixed_thresholds(
        test_scores,
        thresholds
    )

    fixed_metrics.to_csv(
        OUT_ROOT / "fixed_threshold_test_metrics.csv",
        index=False
    )

    print("\nPRIMARY POLICY FIXED-THRESHOLD RESULTS:")
    print(
        fixed_metrics[
            fixed_metrics["policy"] == PRIMARY_POLICY
        ][
            [
                "policy",
                "validation_normal_percentile",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # Dataset-wise
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOMAIN-WISE TEST RESULTS")
    print("=" * 70)

    domain_df = dataset_metrics(
        test_scores
    )

    domain_df.to_csv(
        OUT_ROOT / "domain_wise_test_metrics.csv",
        index=False
    )

    primary_domain = domain_df[
        domain_df["policy"] == PRIMARY_POLICY
    ]

    print(
        primary_domain[
            [
                "dataset",
                "N",
                "normal",
                "anthropogenic",
                "ROC_AUC",
                "PR_AUC",
                "oracle_best_F1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # Signal distributions
    # --------------------------------------------------------

    distribution_rows = []

    for policy in POLICIES:

        for label_value, label_name in [
            (0, "normal"),
            (1, "anthropogenic"),
        ]:

            subset = test_scores[
                test_scores["label"] == label_value
            ]

            x = subset[policy].values

            distribution_rows.append({
                "policy": policy,
                "class": label_name,
                "N": len(x),
                "mean": np.mean(x),
                "median": np.median(x),
                "P90": np.percentile(x, 90),
                "P95": np.percentile(x, 95),
                "P99": np.percentile(x, 99),
            })

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    distribution_df.to_csv(
        OUT_ROOT / "test_evidence_distributions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Curves
    # --------------------------------------------------------

    print("\nSaving ROC / PR curves...")

    save_curves(
        test_scores
    )

    # --------------------------------------------------------
    # Save normalization parameters
    # --------------------------------------------------------

    normalization_rows = []

    for name, p in normalization.items():

        normalization_rows.append({
            "signal": name,
            "source": "validation_normal_only",
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            "P5": p["p5"],
            "P95": p["p95"],
            "raw_column": p["column"],
        })

    pd.DataFrame(
        normalization_rows
    ).to_csv(
        OUT_ROOT / "final_normalization_parameters.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    verdict = make_verdict(
        test_metrics,
        fixed_metrics,
        test_scores,
        normalization
    )

    print("\n" + "=" * 70)
    print(verdict)
    print("=" * 70)

    # --------------------------------------------------------
    # Final file list
    # --------------------------------------------------------

    print("\nOUTPUT FILES")
    print("-" * 70)

    for p in sorted(OUT_ROOT.iterdir()):

        if p.is_file():
            print(
                f"{p.name:45s} "
                f"{p.stat().st_size / 1024:.1f} KB"
            )

    print("\nDONE.")


if __name__ == "__main__":
    main()