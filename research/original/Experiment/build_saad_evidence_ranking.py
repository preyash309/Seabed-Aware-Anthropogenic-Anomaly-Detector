# ============================================================
# SAAD — EVIDENCE RANKING EVALUATION
# ============================================================
#
# Purpose:
#   Determine how the three independent SAAD signals should
#   be combined WITHOUT learning a potentially unstable
#   universal fusion probability.
#
# Signals:
#   YOLO = supervised object evidence
#   VAE  = reconstruction anomaly evidence
#   Flow = latent normality deviation
#
# Candidate evidence policies:
#
#   1. YOLO
#   2. VAE
#   3. Flow
#   4. MAX
#      strongest evidence from any branch
#
#   5. MEAN
#      average evidence from all branches
#
#   6. TOP2_MEAN
#      average of the two strongest branches
#
#   7. YOLO_FLOW_MEAN
#      average of YOLO + Flow
#
#   8. YOLO_VAE_MEAN
#   9. VAE_FLOW_MEAN
#
# Normalization:
#   Each signal is normalized using ONLY NORMAL validation
#   samples, using P5/P95.
#
# IMPORTANT:
#   - Validation set only
#   - Final test set is NOT accessed
#   - No model retraining
#   - No learned meta-classifier
#   - No arbitrary 0.33 / 0.66 thresholds
#
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

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
    r"\saad_evidence_ranking"
)


# ============================================================
# UTILITIES
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_metric(metric_fn, y_true, scores):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(
            metric_fn(
                y_true,
                scores,
            )
        )
    except Exception:
        return np.nan


def best_f1(y_true, scores):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    if len(thresholds) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    f1 = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    idx = int(
        np.nanargmax(f1)
    )

    return (
        float(thresholds[idx]),
        float(precision[idx]),
        float(recall[idx]),
        float(f1[idx]),
    )


def evaluate_policy(
    y_true,
    scores,
):

    roc = safe_metric(
        roc_auc_score,
        y_true,
        scores,
    )

    pr = safe_metric(
        average_precision_score,
        y_true,
        scores,
    )

    threshold, precision, recall, f1 = (
        best_f1(
            y_true,
            scores,
        )
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
# NORMAL-BASED PERCENTILE NORMALIZATION
# ============================================================

def fit_normal_percentile(
    scores,
    labels,
):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
    )

    normal = scores[
        labels == 0
    ]

    if len(normal) < 2:

        raise RuntimeError(
            "Not enough normal validation samples "
            "to fit percentile normalization."
        )

    p5 = float(
        np.percentile(
            normal,
            5,
        )
    )

    p95 = float(
        np.percentile(
            normal,
            95,
        )
    )

    if p95 <= p5:
        p95 = p5 + 1e-12

    return {
        "p5": p5,
        "p95": p95,
        "normal_n": len(normal),
    }


def transform_percentile(
    scores,
    params,
):

    p5 = params["p5"]
    p95 = params["p95"]

    x = (
        np.asarray(
            scores,
            dtype=float,
        )
        - p5
    ) / (
        p95
        - p5
        + 1e-12
    )

    return np.clip(
        x,
        0.0,
        1.0,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAAD — EVIDENCE RANKING EVALUATION")
print("=" * 70)

print()
print("Loading:")
print(INPUT_CSV)

if not os.path.exists(INPUT_CSV):

    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

ensure_dir(
    OUTPUT_DIR
)

df = pd.read_csv(
    INPUT_CSV
)

print()
print(
    f"Validation images: {len(df)}"
)


# ============================================================
# FIND COLUMNS
# ============================================================

DATASET_COL = find_column(
    df,
    [
        "dataset",
        "domain",
        "source_dataset",
    ],
)

LABEL_COL = find_column(
    df,
    [
        "label",
        "y_true",
        "target",
        "class",
    ],
)

YOLO_COL = find_column(
    df,
    [
        "yolo_raw",
        "yolo_score",
        "YOLO",
        "yolo",
    ],
)

VAE_COL = find_column(
    df,
    [
        "vae_raw",
        "vae_score",
        "VAE",
        "vae",
    ],
)

FLOW_COL = find_column(
    df,
    [
        "flow_raw",
        "flow_score",
        "Flow",
        "flow",
    ],
)


required = {
    "dataset": DATASET_COL,
    "label": LABEL_COL,
    "yolo": YOLO_COL,
    "vae": VAE_COL,
    "flow": FLOW_COL,
}

missing = [
    k
    for k, v in required.items()
    if v is None
]

if missing:

    print()
    print(
        "ERROR: Missing required columns:"
    )

    for x in missing:
        print(
            "  ",
            x,
        )

    print()
    print(
        "Available columns:"
    )

    for c in df.columns:
        print(
            "  ",
            c,
        )

    raise RuntimeError(
        "Required columns unavailable."
    )


print()
print("Detected columns:")

print(
    "  Dataset:",
    DATASET_COL,
)

print(
    "  Label:",
    LABEL_COL,
)

print(
    "  YOLO:",
    YOLO_COL,
)

print(
    "  VAE:",
    VAE_COL,
)

print(
    "  Flow:",
    FLOW_COL,
)


# ============================================================
# CLEAN
# ============================================================

work = pd.DataFrame({

    "dataset":
        df[DATASET_COL].astype(str),

    "label":
        pd.to_numeric(
            df[LABEL_COL],
            errors="coerce",
        ),

    "yolo":
        pd.to_numeric(
            df[YOLO_COL],
            errors="coerce",
        ),

    "vae":
        pd.to_numeric(
            df[VAE_COL],
            errors="coerce",
        ),

    "flow":
        pd.to_numeric(
            df[FLOW_COL],
            errors="coerce",
        ),
})


# Preserve useful identifier columns if available.

for candidate in [
    "image",
    "image_path",
    "output_image",
    "filename",
    "path",
    "original_path",
]:

    if candidate in df.columns:

        work[
            candidate
        ] = df[
            candidate
        ]

        break


before = len(work)

work = work.dropna(
    subset=[
        "dataset",
        "label",
        "yolo",
        "vae",
        "flow",
    ]
).reset_index(
    drop=True
)

work["label"] = (
    work["label"]
    .astype(int)
)

after = len(work)

if before != after:

    print()
    print(
        f"Removed {before - after} invalid rows."
    )


labels = work[
    "label"
].values

datasets = work[
    "dataset"
].values


print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    work[
        "label"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("=" * 70)
print("DOMAIN DISTRIBUTION")
print("=" * 70)

domain_table = (
    work
    .groupby("dataset")
    .agg(
        images=("label", "size"),
        normal=(
            "label",
            lambda x:
                int((x == 0).sum()),
        ),
        anthropogenic=(
            "label",
            lambda x:
                int((x == 1).sum()),
        ),
    )
    .reset_index()
)

print(
    domain_table.to_string(
        index=False
    )
)


# ============================================================
# RAW SIGNALS
# ============================================================

raw = {

    "YOLO":
        work[
            "yolo"
        ].values.astype(float),

    "VAE":
        work[
            "vae"
        ].values.astype(float),

    "Flow":
        work[
            "flow"
        ].values.astype(float),
}


# ============================================================
# FIT NORMALIZATION
# ============================================================

print()
print("=" * 70)
print("NORMAL-BASED SIGNAL NORMALIZATION")
print("=" * 70)

normalization = {}

evidence = {}

for name in [
    "YOLO",
    "VAE",
    "Flow",
]:

    params = fit_normal_percentile(
        raw[name],
        labels,
    )

    normalization[
        name
    ] = params

    evidence[
        name
    ] = transform_percentile(
        raw[name],
        params,
    )

    print()
    print(name)

    print(
        f"  Normal N : {params['normal_n']}"
    )

    print(
        f"  P5       : {params['p5']:.8f}"
    )

    print(
        f"  P95      : {params['p95']:.8f}"
    )


# ============================================================
# BUILD EVIDENCE MATRIX
# ============================================================

Y = evidence["YOLO"]
V = evidence["VAE"]
F = evidence["Flow"]

matrix = np.column_stack(
    [
        Y,
        V,
        F,
    ]
)


# ============================================================
# EVIDENCE POLICIES
# ============================================================

policies = {

    # Individual branches
    "YOLO":
        Y,

    "VAE":
        V,

    "Flow":
        F,

    # Strongest branch
    "MAX":
        np.max(
            matrix,
            axis=1,
        ),

    # All three
    "MEAN":
        np.mean(
            matrix,
            axis=1,
        ),

    # Two strongest branches
    "TOP2_MEAN":
        (
            np.sort(
                matrix,
                axis=1,
            )[:, -1]
            +
            np.sort(
                matrix,
                axis=1,
            )[:, -2]
        )
        / 2.0,

    # Specific combinations
    "YOLO_FLOW_MEAN":
        (
            Y + F
        ) / 2.0,

    "YOLO_VAE_MEAN":
        (
            Y + V
        ) / 2.0,

    "VAE_FLOW_MEAN":
        (
            V + F
        ) / 2.0,
}


# ============================================================
# ADDITIONAL AGREEMENT FEATURES
# ============================================================

sorted_evidence = np.sort(
    matrix,
    axis=1,
)

second_highest = (
    sorted_evidence[
        :,
        -2,
    ]
)

lowest = (
    sorted_evidence[
        :,
        0,
    ]
)

highest = (
    sorted_evidence[
        :,
        -1,
    ]
)

mean_evidence = np.mean(
    matrix,
    axis=1,
)

evidence_spread = (
    highest - lowest
)

evidence_std = np.std(
    matrix,
    axis=1,
)


policies[
    "SECOND_HIGHEST"
] = second_highest

policies[
    "LOWEST"
] = lowest

policies[
    "EVIDENCE_STD"
] = evidence_std


# ============================================================
# EVALUATE OVERALL
# ============================================================

print()
print("=" * 70)
print("OVERALL EVIDENCE POLICY PERFORMANCE")
print("=" * 70)

overall_rows = []

for policy_name, scores in policies.items():

    # EVIDENCE_STD is different:
    # high spread means disagreement, not anomaly.
    # Reverse it so higher score means stronger
    # agreement / consistency.
    if policy_name == "EVIDENCE_STD":

        scores_for_eval = (
            1.0 - np.clip(
                scores,
                0.0,
                1.0,
            )
        )

    else:

        scores_for_eval = scores

    metrics = evaluate_policy(
        labels,
        scores_for_eval,
    )

    overall_rows.append({

        "policy":
            policy_name,

        **metrics,
    })


overall_df = pd.DataFrame(
    overall_rows
)

overall_df = overall_df.sort_values(
    "PR_AUC",
    ascending=False,
)

print(
    overall_df.to_string(
        index=False
    )
)

overall_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_policy_results.csv",
    ),
    index=False,
)


# ============================================================
# DATASET-WISE EVALUATION
# ============================================================

print()
print("=" * 70)
print("DATASET-WISE PERFORMANCE")
print("=" * 70)

domain_results = []

for dataset_name in sorted(
    work["dataset"].unique()
):

    mask = (
        work["dataset"].values
        == dataset_name
    )

    y_domain = labels[
        mask
    ]

    print()
    print(
        "-" * 70
    )

    print(
        f"DOMAIN: {dataset_name}"
    )

    print(
        f"N = {len(y_domain)}"
    )

    print(
        "Class counts:",
        {
            0: int(
                (y_domain == 0).sum()
            ),
            1: int(
                (y_domain == 1).sum()
            ),
        }
    )

    for policy_name, scores in policies.items():

        if policy_name == "EVIDENCE_STD":

            scores_eval = (
                1.0 - np.clip(
                    scores,
                    0.0,
                    1.0,
                )
            )

        else:

            scores_eval = scores

        metrics = evaluate_policy(
            y_domain,
            scores_eval[
                mask
            ],
        )

        row = {

            "dataset":
                dataset_name,

            "policy":
                policy_name,

            "N":
                len(y_domain),

            "normal":
                int(
                    (y_domain == 0).sum()
                ),

            "anthropogenic":
                int(
                    (y_domain == 1).sum()
                ),

            **metrics,
        }

        domain_results.append(
            row
        )


domain_df = pd.DataFrame(
    domain_results
)

domain_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_wise_policy_results.csv",
    ),
    index=False,
)


# ============================================================
# PRINT DOMAIN TABLE
# ============================================================

for dataset_name in sorted(
    work["dataset"].unique()
):

    temp = domain_df[
        domain_df[
            "dataset"
        ]
        == dataset_name
    ].copy()

    temp = temp.sort_values(
        "PR_AUC",
        ascending=False,
    )

    print()
    print(
        f"{dataset_name}"
    )

    print(
        temp[
            [
                "policy",
                "ROC_AUC",
                "PR_AUC",
                "Best_F1",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# NORMAL VS ANTHROPOGENIC SCORE DISTRIBUTIONS
# ============================================================

print()
print("=" * 70)
print("CLASS-WISE EVIDENCE DISTRIBUTIONS")
print("=" * 70)

distribution_rows = []

for policy_name, scores in policies.items():

    if policy_name == "EVIDENCE_STD":

        scores_eval = (
            1.0 - np.clip(
                scores,
                0.0,
                1.0,
            )
        )

    else:

        scores_eval = scores

    normal = scores_eval[
        labels == 0
    ]

    anthropogenic = scores_eval[
        labels == 1
    ]

    distribution_rows.append({

        "policy":
            policy_name,

        "normal_N":
            len(normal),

        "normal_mean":
            float(np.mean(normal)),

        "normal_median":
            float(np.median(normal)),

        "normal_P90":
            float(
                np.percentile(
                    normal,
                    90,
                )
            ),

        "normal_P95":
            float(
                np.percentile(
                    normal,
                    95,
                )
            ),

        "normal_P99":
            float(
                np.percentile(
                    normal,
                    99,
                )
            ),

        "anthropogenic_N":
            len(anthropogenic),

        "anthropogenic_mean":
            float(
                np.mean(
                    anthropogenic
                )
            ),

        "anthropogenic_median":
            float(
                np.median(
                    anthropogenic
                )
            ),

        "anthropogenic_P90":
            float(
                np.percentile(
                    anthropogenic,
                    90,
                )
            ),

        "anthropogenic_P95":
            float(
                np.percentile(
                    anthropogenic,
                    95,
                )
            ),

        "anthropogenic_P99":
            float(
                np.percentile(
                    anthropogenic,
                    99,
                )
            ),
    })


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_wise_distributions.csv",
    ),
    index=False,
)

print(
    distribution_df.to_string(
        index=False
    )
)


# ============================================================
# PER-IMAGE EVIDENCE TABLE
# ============================================================

per_image = work.copy()

per_image[
    "yolo_evidence"
] = Y

per_image[
    "vae_evidence"
] = V

per_image[
    "flow_evidence"
] = F

per_image[
    "max_evidence"
] = highest

per_image[
    "second_highest_evidence"
] = second_highest

per_image[
    "lowest_evidence"
] = lowest

per_image[
    "mean_evidence"
] = mean_evidence

per_image[
    "evidence_spread"
] = evidence_spread

per_image[
    "evidence_std"
] = evidence_std


# ============================================================
# INTERPRETATION LABELS
# ============================================================

def evidence_profile(
    y,
    v,
    f,
):

    values = {
        "YOLO": y,
        "VAE": v,
        "Flow": f,
    }

    ordered = sorted(
        values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    strongest_name = ordered[0][0]
    second_name = ordered[1][0]

    strongest = ordered[0][1]
    second = ordered[1][1]

    # --------------------------------------------------------
    # These are descriptive profiles, NOT hard decisions.
    # --------------------------------------------------------

    if (
        strongest >= 0.75
        and second >= 0.60
    ):

        profile = (
            "STRONG_MULTI_SIGNAL"
        )

    elif (
        strongest >= 0.75
        and second < 0.40
    ):

        profile = (
            "SINGLE_SIGNAL_NOVELTY"
        )

    elif (
        strongest >= 0.60
        and second >= 0.40
    ):

        profile = (
            "MODERATE_CORROBORATION"
        )

    elif strongest < 0.40:

        profile = (
            "LOW_EVIDENCE"
        )

    else:

        profile = (
            "AMBIGUOUS"
        )

    return (
        profile,
        strongest_name,
        second_name,
    )


profiles = [
    evidence_profile(
        y,
        v,
        f,
    )
    for y, v, f in zip(
        Y,
        V,
        F,
    )
]

per_image[
    "evidence_profile"
] = [
    x[0]
    for x in profiles
]

per_image[
    "strongest_signal"
] = [
    x[1]
    for x in profiles
]

per_image[
    "second_strongest_signal"
] = [
    x[2]
    for x in profiles
]


# ============================================================
# RANKING
# ============================================================

# Primary candidate ranking:
#   first = strongest evidence
#   second = corroboration

per_image = per_image.sort_values(
    [
        "max_evidence",
        "second_highest_evidence",
    ],
    ascending=[
        False,
        False,
    ],
)

per_image[
    "evidence_rank"
] = np.arange(
    1,
    len(per_image) + 1,
)


per_image.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_image_evidence_ranking.csv",
    ),
    index=False,
)


# ============================================================
# PROFILE SUMMARY
# ============================================================

profile_summary = (
    per_image
    .groupby(
        [
            "label",
            "evidence_profile",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

profile_summary[
    "class"
] = profile_summary[
    "label"
].map(
    {
        0: "NORMAL",
        1: "ANTHROPOGENIC",
    }
)

profile_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "evidence_profile_summary.csv",
    ),
    index=False,
)


# ============================================================
# SIGNAL AGREEMENT MATRIX
# ============================================================

# Count how often each pair is simultaneously strong.
#
# Strong = >= normal P95 of that signal.
#
# This is descriptive only.

strong_masks = {}

for signal_name in [
    "YOLO",
    "VAE",
    "Flow",
]:

    normal_signal = evidence[
        signal_name
    ][
        labels == 0
    ]

    threshold = float(
        np.percentile(
            normal_signal,
            95,
        )
    )

    strong_masks[
        signal_name
    ] = evidence[
        signal_name
    ] >= threshold


agreement_rows = []

for dataset_name in [
    "ALL",
    *sorted(
        work["dataset"].unique()
    ),
]:

    if dataset_name == "ALL":

        mask = np.ones(
            len(work),
            dtype=bool,
        )

    else:

        mask = (
            work["dataset"].values
            == dataset_name
        )

    subset_n = int(
        mask.sum()
    )

    if subset_n == 0:
        continue

    y_mask = strong_masks[
        "YOLO"
    ][mask]

    v_mask = strong_masks[
        "VAE"
    ][mask]

    f_mask = strong_masks[
        "Flow"
    ][mask]

    agreement_rows.append({

        "dataset":
            dataset_name,

        "N":
            subset_n,

        "YOLO_strong_rate":
            float(
                y_mask.mean()
            ),

        "VAE_strong_rate":
            float(
                v_mask.mean()
            ),

        "Flow_strong_rate":
            float(
                f_mask.mean()
            ),

        "YOLO_VAE_both":
            float(
                (
                    y_mask
                    & v_mask
                ).mean()
            ),

        "YOLO_Flow_both":
            float(
                (
                    y_mask
                    & f_mask
                ).mean()
            ),

        "VAE_Flow_both":
            float(
                (
                    v_mask
                    & f_mask
                ).mean()
            ),

        "ALL_THREE":
            float(
                (
                    y_mask
                    & v_mask
                    & f_mask
                ).mean()
            ),

        "AT_LEAST_TWO":
            float(
                (
                    y_mask.astype(int)
                    + v_mask.astype(int)
                    + f_mask.astype(int)
                    >= 2
                ).mean()
            ),
    })


agreement_df = pd.DataFrame(
    agreement_rows
)

agreement_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "signal_agreement_summary.csv",
    ),
    index=False,
)


# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

normalization_df = pd.DataFrame([
    {
        "signal": name,
        **params,
    }
    for name, params
    in normalization.items()
])

normalization_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normalization_parameters.csv",
    ),
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "evidence_ranking_summary.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SAAD — EVIDENCE RANKING EVALUATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "This experiment uses ONLY the validation set.\n"
    )

    f.write(
        "The final test set was NOT accessed.\n"
    )

    f.write(
        "No learned meta-classifier was used.\n"
    )

    f.write(
        "No arbitrary probability threshold was imposed.\n\n"
    )

    f.write(
        f"Validation images: {len(work)}\n"
    )

    f.write(
        f"Normal: {int((labels == 0).sum())}\n"
    )

    f.write(
        f"Anthropogenic: {int((labels == 1).sum())}\n\n"
    )


    f.write(
        "OVERALL POLICY PERFORMANCE\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        overall_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "CLASS-WISE DISTRIBUTIONS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        distribution_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "SIGNAL AGREEMENT\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        agreement_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        "MAX asks whether ANY evidence branch is strongly "
        "suspicious.\n"
    )

    f.write(
        "TOP2_MEAN asks whether TWO independent branches "
        "support the anomaly.\n"
    )

    f.write(
        "MEAN asks whether ALL THREE branches collectively "
        "support the anomaly.\n"
    )

    f.write(
        "YOLO_FLOW_MEAN tests the most stable pair suggested "
        "by the previous experiments.\n"
    )

    f.write(
        "EVIDENCE_STD is converted to 1-STD for ranking "
        "evaluation, so higher means greater agreement.\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "The evidence profiles are descriptive and should "
        "not yet be interpreted as deployment thresholds.\n"
    )

    f.write(
        "The final test set must remain untouched until "
        "the evaluation protocol is frozen.\n"
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("TOP POLICIES BY PR-AUC")
print("=" * 70)

print(
    overall_df[
        [
            "policy",
            "ROC_AUC",
            "PR_AUC",
            "Best_F1",
            "Best_F1_precision",
            "Best_F1_recall",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print()
print("=" * 70)
print("SIGNAL AGREEMENT")
print("=" * 70)

print(
    agreement_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("EVIDENCE PROFILE COUNTS")
print("=" * 70)

print(
    profile_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("TOP 30 RANKED CANDIDATES")
print("=" * 70)

candidate_columns = [
    "dataset",
    "label",
    "yolo_evidence",
    "vae_evidence",
    "flow_evidence",
    "max_evidence",
    "second_highest_evidence",
    "mean_evidence",
    "evidence_spread",
    "strongest_signal",
    "second_strongest_signal",
    "evidence_profile",
]

candidate_columns = [
    c
    for c in candidate_columns
    if c in per_image.columns
]

print(
    per_image[
        candidate_columns
    ]
    .head(30)
    .to_string(
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
    "  overall_policy_results.csv"
)

print(
    "  dataset_wise_policy_results.csv"
)

print(
    "  class_wise_distributions.csv"
)

print(
    "  per_image_evidence_ranking.csv"
)

print(
    "  evidence_profile_summary.csv"
)

print(
    "  signal_agreement_summary.csv"
)

print(
    "  normalization_parameters.csv"
)

print(
    "  evidence_ranking_summary.txt"
)