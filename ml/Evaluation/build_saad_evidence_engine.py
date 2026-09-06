# ============================================================
# SAAD — EVIDENCE & UNCERTAINTY ENGINE
# ============================================================
#
# Purpose:
#   Convert the existing YOLO / VAE / Flow raw scores into a
#   transparent evidence-agreement decision system.
#
# IMPORTANT:
#   - Uses ONLY validation_raw_scores.csv
#   - Does NOT access the final 544-image test set
#   - Does NOT retrain YOLO, VAE or Flow
#   - Does NOT optimize a fixed fusion weight
#
# Primary calibration:
#   Platt
#
# Comparison:
#   Percentile
#   Platt
#   Isotonic
#
# Evidence signals:
#   YOLO       = supervised object evidence
#   VAE        = reconstruction/anomaly evidence
#   Flow       = latent normality evidence
#
# Derived quantities:
#
#   mean_evidence
#   max_evidence
#   min_evidence
#   evidence_std
#   agreement
#   disagreement
#
# Decision:
#
#   NORMAL
#   REVIEW
#   ANOMALY
#
# Thresholds are derived ONLY from validation data.
#
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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
    r"\\saad_evidence_engine"
)

RANDOM_STATE = 42


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.percentile(x, q))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.median(x))


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# CALIBRATION
# ============================================================

def percentile_fit(scores, labels):
    """
    Fit percentile normalization using NORMAL samples only.
    """

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    normal = scores[labels == 0]

    if len(normal) < 2:
        p5 = float(np.min(scores))
        p95 = float(np.max(scores))
    else:
        p5 = float(np.percentile(normal, 5))
        p95 = float(np.percentile(normal, 95))

    if p95 <= p5:
        p95 = p5 + 1e-12

    return p5, p95


def percentile_transform(scores, params):
    p5, p95 = params

    x = (
        np.asarray(scores, dtype=float) - p5
    ) / (
        p95 - p5 + 1e-12
    )

    return np.clip(x, 0.0, 1.0)


def platt_fit(scores, labels):
    """
    Balanced logistic calibration.
    """

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )

    model.fit(
        np.asarray(scores).reshape(-1, 1),
        labels,
    )

    return model


def platt_transform(scores, model):
    return model.predict_proba(
        np.asarray(scores).reshape(-1, 1)
    )[:, 1]


def isotonic_fit(scores, labels):

    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(
        np.asarray(scores, dtype=float),
        labels,
    )

    return model


def isotonic_transform(scores, model):

    return model.predict(
        np.asarray(scores, dtype=float)
    )


# ============================================================
# EVIDENCE FEATURES
# ============================================================

def calculate_evidence_features(
    yolo,
    vae,
    flow,
):
    """
    All inputs must already be calibrated into [0,1].

    Higher = stronger anomaly evidence.
    """

    matrix = np.column_stack(
        [
            yolo,
            vae,
            flow,
        ]
    )

    mean_evidence = np.mean(
        matrix,
        axis=1,
    )

    max_evidence = np.max(
        matrix,
        axis=1,
    )

    min_evidence = np.min(
        matrix,
        axis=1,
    )

    evidence_std = np.std(
        matrix,
        axis=1,
    )

    # Agreement is high when all three signals
    # are simultaneously strong.
    #
    # Geometric mean is deliberately used here.
    # If one signal is near zero, agreement drops.
    agreement = np.power(
        np.clip(
            yolo * vae * flow,
            0.0,
            1.0,
        ),
        1.0 / 3.0,
    )

    # Disagreement is simply normalized spread.
    disagreement = np.clip(
        evidence_std,
        0.0,
        1.0,
    )

    return {
        "mean_evidence": mean_evidence,
        "max_evidence": max_evidence,
        "min_evidence": min_evidence,
        "evidence_std": evidence_std,
        "agreement": agreement,
        "disagreement": disagreement,
    }


# ============================================================
# DECISION THRESHOLDS
# ============================================================

def derive_thresholds(
    df,
    score_column,
):
    """
    Thresholds are derived from NORMAL validation examples.

    NORMAL:
        below normal P95

    REVIEW:
        normal P95 -> normal P99

    ANOMALY:
        above normal P99

    IMPORTANT:
        This is deliberately conservative.
        It does NOT use labeled anthropogenic examples
        to choose deployment thresholds.
    """

    normal_scores = df.loc[
        df["label"] == 0,
        score_column,
    ].values

    p95 = safe_percentile(
        normal_scores,
        95,
    )

    p99 = safe_percentile(
        normal_scores,
        99,
    )

    return {
        "normal_p95": p95,
        "normal_p99": p99,
    }


def classify_score(
    score,
    thresholds,
):
    p95 = thresholds["normal_p95"]
    p99 = thresholds["normal_p99"]

    if score < p95:
        return "NORMAL"

    elif score < p99:
        return "REVIEW"

    else:
        return "ANOMALY"


# ============================================================
# CONFIDENCE / EVIDENCE CATEGORY
# ============================================================

def evidence_category(
    yolo,
    vae,
    flow,
):
    """
    Categorize each evidence source.

    < 0.33 = LOW
    0.33-0.66 = MEDIUM
    >= 0.66 = HIGH
    """

    def category(x):
        if x < 0.33:
            return "LOW"

        if x < 0.66:
            return "MEDIUM"

        return "HIGH"

    return (
        category(yolo),
        category(vae),
        category(flow),
    )


# ============================================================
# AGREEMENT-BASED DECISION
# ============================================================

def agreement_decision(
    yolo,
    vae,
    flow,
):
    """
    Transparent evidence logic.

    HIGH AGREEMENT:
        at least two signals HIGH
        and remaining signal >= MEDIUM

    REVIEW:
        evidence disagreement
        or one strong signal without corroboration

    NORMAL:
        no strong anomaly evidence
    """

    values = np.array(
        [
            yolo,
            vae,
            flow,
        ],
        dtype=float,
    )

    high = values >= 0.66
    medium = values >= 0.33

    n_high = int(high.sum())
    n_medium = int(medium.sum())

    if (
        n_high >= 2
        and n_medium == 3
    ):
        return "ANOMALY"

    if n_high >= 1:
        return "REVIEW"

    return "NORMAL"


# ============================================================
# AGREEMENT SCORE
# ============================================================

def agreement_strength(
    yolo,
    vae,
    flow,
):
    """
    Continuous agreement measure.

    Geometric mean gives high agreement only when
    all signals are high.

    Pairwise agreement adds robustness.
    """

    vals = np.array(
        [
            yolo,
            vae,
            flow,
        ],
        dtype=float,
    )

    geometric = np.prod(
        np.clip(vals, 0.0, 1.0)
    ) ** (1.0 / 3.0)

    pairwise = np.mean(
        [
            min(yolo, vae),
            min(yolo, flow),
            min(vae, flow),
        ]
    )

    return float(
        0.5 * geometric
        + 0.5 * pairwise
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SAAD — EVIDENCE & UNCERTAINTY ENGINE")
print("=" * 70)

print()
print("Input:")
print(INPUT_CSV)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

ensure_dir(OUTPUT_DIR)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print()
print(
    f"Validation images: {len(df)}"
)


# ============================================================
# DETECT COLUMNS
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
            " ",
            x,
        )

    print()
    print(
        "Available columns:"
    )

    for c in df.columns:
        print(
            " ",
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
# STANDARDIZE
# ============================================================

work = df.copy()

work["dataset_clean"] = work[
    DATASET_COL
].astype(str)

work["label_clean"] = (
    pd.to_numeric(
        work[LABEL_COL],
        errors="coerce",
    )
    .astype("Int64")
)

work["yolo_clean"] = pd.to_numeric(
    work[YOLO_COL],
    errors="coerce",
)

work["vae_clean"] = pd.to_numeric(
    work[VAE_COL],
    errors="coerce",
)

work["flow_clean"] = pd.to_numeric(
    work[FLOW_COL],
    errors="coerce",
)

work = work.dropna(
    subset=[
        "dataset_clean",
        "label_clean",
        "yolo_clean",
        "vae_clean",
        "flow_clean",
    ]
).copy()

work["label_clean"] = (
    work["label_clean"]
    .astype(int)
)

print()
print("Class distribution:")

print(
    work["label_clean"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# RAW ARRAYS
# ============================================================

labels = work[
    "label_clean"
].values

yolo_raw = work[
    "yolo_clean"
].values.astype(float)

vae_raw = work[
    "vae_clean"
].values.astype(float)

flow_raw = work[
    "flow_clean"
].values.astype(float)


# ============================================================
# CALIBRATION
# ============================================================

print()
print("=" * 70)
print("FITTING CALIBRATION")
print("=" * 70)


# ------------------------------------------------------------
# Percentile
# ------------------------------------------------------------

pct_yolo_params = percentile_fit(
    yolo_raw,
    labels,
)

pct_vae_params = percentile_fit(
    vae_raw,
    labels,
)

pct_flow_params = percentile_fit(
    flow_raw,
    labels,
)

pct_yolo = percentile_transform(
    yolo_raw,
    pct_yolo_params,
)

pct_vae = percentile_transform(
    vae_raw,
    pct_vae_params,
)

pct_flow = percentile_transform(
    flow_raw,
    pct_flow_params,
)


# ------------------------------------------------------------
# Platt
# ------------------------------------------------------------

platt_yolo_model = platt_fit(
    yolo_raw,
    labels,
)

platt_vae_model = platt_fit(
    vae_raw,
    labels,
)

platt_flow_model = platt_fit(
    flow_raw,
    labels,
)

platt_yolo = platt_transform(
    yolo_raw,
    platt_yolo_model,
)

platt_vae = platt_transform(
    vae_raw,
    platt_vae_model,
)

platt_flow = platt_transform(
    flow_raw,
    platt_flow_model,
)


# ------------------------------------------------------------
# Isotonic
# ------------------------------------------------------------

iso_yolo_model = isotonic_fit(
    yolo_raw,
    labels,
)

iso_vae_model = isotonic_fit(
    vae_raw,
    labels,
)

iso_flow_model = isotonic_fit(
    flow_raw,
    labels,
)

iso_yolo = isotonic_transform(
    yolo_raw,
    iso_yolo_model,
)

iso_vae = isotonic_transform(
    vae_raw,
    iso_vae_model,
)

iso_flow = isotonic_transform(
    flow_raw,
    iso_flow_model,
)


# ============================================================
# BUILD CALIBRATION DATASETS
# ============================================================

calibrations = {

    "percentile": {
        "yolo": pct_yolo,
        "vae": pct_vae,
        "flow": pct_flow,
    },

    "platt": {
        "yolo": platt_yolo,
        "vae": platt_vae,
        "flow": platt_flow,
    },

    "isotonic": {
        "yolo": iso_yolo,
        "vae": iso_vae,
        "flow": iso_flow,
    },
}


# ============================================================
# BUILD EVIDENCE TABLES
# ============================================================

all_tables = {}

for calibration_name, signals in calibrations.items():

    features = calculate_evidence_features(
        signals["yolo"],
        signals["vae"],
        signals["flow"],
    )

    out = pd.DataFrame({
        "dataset": work[
            "dataset_clean"
        ].values,

        "label": labels,

        "yolo_evidence": signals[
            "yolo"
        ],

        "vae_evidence": signals[
            "vae"
        ],

        "flow_evidence": signals[
            "flow"
        ],

        "mean_evidence": features[
            "mean_evidence"
        ],

        "max_evidence": features[
            "max_evidence"
        ],

        "min_evidence": features[
            "min_evidence"
        ],

        "evidence_std": features[
            "evidence_std"
        ],

        "agreement": features[
            "agreement"
        ],

        "disagreement": features[
            "disagreement"
        ],
    })

    # --------------------------------------------------------
    # Strongest evidence source
    # --------------------------------------------------------

    evidence_matrix = out[
        [
            "yolo_evidence",
            "vae_evidence",
            "flow_evidence",
        ]
    ].values

    source_idx = np.argmax(
        evidence_matrix,
        axis=1,
    )

    source_names = np.array(
        [
            "YOLO",
            "VAE",
            "Flow",
        ]
    )

    out["strongest_evidence"] = (
        source_names[source_idx]
    )


    # --------------------------------------------------------
    # Evidence categories
    # --------------------------------------------------------

    categories = [
        evidence_category(
            y,
            v,
            f,
        )
        for y, v, f in zip(
            out["yolo_evidence"],
            out["vae_evidence"],
            out["flow_evidence"],
        )
    ]

    out[
        "yolo_level"
    ] = [
        x[0]
        for x in categories
    ]

    out[
        "vae_level"
    ] = [
        x[1]
        for x in categories
    ]

    out[
        "flow_level"
    ] = [
        x[2]
        for x in categories
    ]


    # --------------------------------------------------------
    # Agreement decision
    # --------------------------------------------------------

    out[
        "agreement_decision"
    ] = [
        agreement_decision(
            y,
            v,
            f,
        )
        for y, v, f in zip(
            out["yolo_evidence"],
            out["vae_evidence"],
            out["flow_evidence"],
        )
    ]


    # --------------------------------------------------------
    # Continuous agreement
    # --------------------------------------------------------

    out[
        "agreement_strength"
    ] = [
        agreement_strength(
            y,
            v,
            f,
        )
        for y, v, f in zip(
            out["yolo_evidence"],
            out["vae_evidence"],
            out["flow_evidence"],
        )
    ]


    # --------------------------------------------------------
    # Normal-distribution threshold decision
    #
    # This is based on mean evidence.
    # --------------------------------------------------------

    threshold_params = derive_thresholds(
        out,
        "mean_evidence",
    )

    out[
        "threshold_decision"
    ] = [
        classify_score(
            x,
            threshold_params,
        )
        for x in out[
            "mean_evidence"
        ]
    ]


    # --------------------------------------------------------
    # Conservative combined decision
    #
    # If agreement logic says ANOMALY:
    #     ANOMALY
    #
    # Else if normal-distribution threshold says ANOMALY:
    #     REVIEW
    #
    # Else if either says REVIEW:
    #     REVIEW
    #
    # Else:
    #     NORMAL
    # --------------------------------------------------------

    decisions = []

    for agreement_dec, threshold_dec in zip(
        out[
            "agreement_decision"
        ],
        out[
            "threshold_decision"
        ],
    ):

        if agreement_dec == "ANOMALY":

            final = "ANOMALY"

        elif threshold_dec == "ANOMALY":

            final = "REVIEW"

        elif (
            agreement_dec == "REVIEW"
            or threshold_dec == "REVIEW"
        ):

            final = "REVIEW"

        else:

            final = "NORMAL"

        decisions.append(final)

    out[
        "saad_decision"
    ] = decisions


    # --------------------------------------------------------
    # Decision confidence
    # --------------------------------------------------------

    confidence = []

    for _, row in out.iterrows():

        y = row[
            "yolo_evidence"
        ]

        v = row[
            "vae_evidence"
        ]

        f = row[
            "flow_evidence"
        ]

        decision = row[
            "saad_decision"
        ]

        if decision == "ANOMALY":

            # Agreement is especially important
            # for an automatic anomaly decision.
            c = (
                0.5 * row[
                    "agreement_strength"
                ]
                + 0.5 * row[
                    "mean_evidence"
                ]
            )

        elif decision == "NORMAL":

            # For normal cases, low evidence across
            # all signals should give high confidence.
            c = 1.0 - row[
                "max_evidence"
            ]

        else:

            # Review cases are intrinsically uncertain.
            # Confidence is deliberately capped.
            c = 0.5 * (
                1.0
                - row[
                    "disagreement"
                ]
            )

        confidence.append(
            float(
                np.clip(
                    c,
                    0.0,
                    1.0,
                )
            )
        )

    out[
        "decision_confidence"
    ] = confidence


    # --------------------------------------------------------
    # Preserve source filename/path if available
    # --------------------------------------------------------

    for candidate in [
        "image",
        "image_path",
        "output_image",
        "filename",
        "path",
        "original_path",
    ]:

        if candidate in work.columns:

            out[
                candidate
            ] = work[
                candidate
            ].values

            break


    all_tables[
        calibration_name
    ] = out


# ============================================================
# PRIMARY CALIBRATION = PLATT
# ============================================================

primary = all_tables[
    "platt"
].copy()


# ============================================================
# THRESHOLD SUMMARY
# ============================================================

threshold_rows = []

for calibration_name, table in all_tables.items():

    thresholds = derive_thresholds(
        table,
        "mean_evidence",
    )

    threshold_rows.append({
        "calibration": calibration_name,
        "normal_mean_evidence_P95": thresholds[
            "normal_p95"
        ],
        "normal_mean_evidence_P99": thresholds[
            "normal_p99"
        ],
    })

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_thresholds.csv",
    ),
    index=False,
)


# ============================================================
# SAVE PER-IMAGE TABLES
# ============================================================

for calibration_name, table in all_tables.items():

    table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"evidence_scores_{calibration_name}.csv",
        ),
        index=False,
    )


primary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "saad_primary_evidence_scores.csv",
    ),
    index=False,
)


# ============================================================
# DATASET-WISE SUMMARY
# ============================================================

summary_rows = []

for calibration_name, table in all_tables.items():

    for dataset_name, group in table.groupby(
        "dataset"
    ):

        summary_rows.append({

            "calibration":
                calibration_name,

            "dataset":
                dataset_name,

            "N":
                len(group),

            "normal":
                int(
                    (group["label"] == 0).sum()
                ),

            "anthropogenic":
                int(
                    (group["label"] == 1).sum()
                ),

            "mean_yolo":
                safe_mean(
                    group["yolo_evidence"]
                ),

            "mean_vae":
                safe_mean(
                    group["vae_evidence"]
                ),

            "mean_flow":
                safe_mean(
                    group["flow_evidence"]
                ),

            "mean_evidence":
                safe_mean(
                    group["mean_evidence"]
                ),

            "mean_agreement":
                safe_mean(
                    group["agreement"]
                ),

            "mean_disagreement":
                safe_mean(
                    group["disagreement"]
                ),

            "anomaly_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "ANOMALY"
                    ).mean()
                ),

            "review_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "REVIEW"
                    ).mean()
                ),

            "normal_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "NORMAL"
                    ).mean()
                ),
        })


dataset_summary = pd.DataFrame(
    summary_rows
)

dataset_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_wise_evidence_summary.csv",
    ),
    index=False,
)


# ============================================================
# CLASS-WISE SUMMARY
# ============================================================

class_rows = []

for calibration_name, table in all_tables.items():

    for label_value, label_name in [
        (0, "NORMAL"),
        (1, "ANTHROPOGENIC"),
    ]:

        group = table[
            table["label"] == label_value
        ]

        if len(group) == 0:
            continue

        class_rows.append({

            "calibration":
                calibration_name,

            "class":
                label_name,

            "N":
                len(group),

            "yolo_mean":
                safe_mean(
                    group["yolo_evidence"]
                ),

            "vae_mean":
                safe_mean(
                    group["vae_evidence"]
                ),

            "flow_mean":
                safe_mean(
                    group["flow_evidence"]
                ),

            "mean_evidence":
                safe_mean(
                    group["mean_evidence"]
                ),

            "median_evidence":
                safe_median(
                    group["mean_evidence"]
                ),

            "agreement":
                safe_mean(
                    group["agreement"]
                ),

            "disagreement":
                safe_mean(
                    group["disagreement"]
                ),
        })


class_summary = pd.DataFrame(
    class_rows
)

class_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_wise_evidence_summary.csv",
    ),
    index=False,
)


# ============================================================
# DECISION SUMMARY
# ============================================================

decision_rows = []

for calibration_name, table in all_tables.items():

    for label_value, label_name in [
        (0, "NORMAL"),
        (1, "ANTHROPOGENIC"),
    ]:

        group = table[
            table["label"] == label_value
        ]

        total = len(group)

        if total == 0:
            continue

        decision_rows.append({

            "calibration":
                calibration_name,

            "class":
                label_name,

            "N":
                total,

            "NORMAL_count":
                int(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "NORMAL"
                    ).sum()
                ),

            "REVIEW_count":
                int(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "REVIEW"
                    ).sum()
                ),

            "ANOMALY_count":
                int(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "ANOMALY"
                    ).sum()
                ),

            "NORMAL_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "NORMAL"
                    ).mean()
                ),

            "REVIEW_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "REVIEW"
                    ).mean()
                ),

            "ANOMALY_rate":
                float(
                    (
                        group[
                            "saad_decision"
                        ]
                        == "ANOMALY"
                    ).mean()
                ),
        })


decision_summary = pd.DataFrame(
    decision_rows
)

decision_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_summary.csv",
    ),
    index=False,
)


# ============================================================
# HIGH-VALUE CANDIDATES
# ============================================================

# Sort by:
#   1. SAAD decision
#   2. agreement
#   3. mean evidence
#
# These are the candidates a human reviewer should inspect
# first in the eventual dashboard.

decision_rank = {
    "ANOMALY": 0,
    "REVIEW": 1,
    "NORMAL": 2,
}

candidate_table = primary.copy()

candidate_table[
    "_decision_rank"
] = candidate_table[
    "saad_decision"
].map(
    decision_rank
)

candidate_table = candidate_table.sort_values(
    [
        "_decision_rank",
        "agreement_strength",
        "mean_evidence",
    ],
    ascending=[
        True,
        False,
        False,
    ],
)

candidate_table = candidate_table.drop(
    columns=[
        "_decision_rank"
    ]
)

candidate_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ranked_review_candidates.csv",
    ),
    index=False,
)


# ============================================================
# UNCERTAINTY TABLE
# ============================================================

uncertain = primary.copy()

# Highest disagreement first
uncertain = uncertain.sort_values(
    [
        "disagreement",
        "mean_evidence",
    ],
    ascending=[
        False,
        False,
    ],
)

uncertain.head(
    min(
        200,
        len(uncertain),
    )
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_200_uncertain_candidates.csv",
    ),
    index=False,
)


# ============================================================
# REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "saad_evidence_engine_summary.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SAAD — EVIDENCE & UNCERTAINTY ENGINE\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "DATA SOURCE\n"
    )

    f.write(
        f"{INPUT_CSV}\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "Only the validation set was used.\n"
    )

    f.write(
        "The final test set was NOT accessed.\n\n"
    )

    f.write(
        f"Validation images: {len(primary)}\n"
    )

    f.write(
        f"Normal: {int((labels == 0).sum())}\n"
    )

    f.write(
        f"Anthropogenic: {int((labels == 1).sum())}\n\n"
    )


    # --------------------------------------------------------
    # Calibration statistics
    # --------------------------------------------------------

    f.write(
        "CALIBRATION COMPARISON\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for name, table in all_tables.items():

        normal = table[
            table["label"] == 0
        ]

        anthropogenic = table[
            table["label"] == 1
        ]

        f.write(
            f"\n{name.upper()}\n"
        )

        f.write(
            f"Normal mean evidence: "
            f"{safe_mean(normal['mean_evidence']):.6f}\n"
        )

        f.write(
            f"Anthropogenic mean evidence: "
            f"{safe_mean(anthropogenic['mean_evidence']):.6f}\n"
        )

        f.write(
            f"Normal median evidence: "
            f"{safe_median(normal['mean_evidence']):.6f}\n"
        )

        f.write(
            f"Anthropogenic median evidence: "
            f"{safe_median(anthropogenic['mean_evidence']):.6f}\n"
        )

        f.write(
            f"Normal mean agreement: "
            f"{safe_mean(normal['agreement']):.6f}\n"
        )

        f.write(
            f"Anthropogenic mean agreement: "
            f"{safe_mean(anthropogenic['agreement']):.6f}\n"
        )


    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    f.write(
        "\n\nDECISION THRESHOLDS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        threshold_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Decision summary
    # --------------------------------------------------------

    f.write(
        "\n\nDECISION SUMMARY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        decision_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    f.write(
        "\n\nDATASET-WISE SUMMARY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        dataset_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Methodological note
    # --------------------------------------------------------

    f.write(
        "\n\nMETHODOLOGICAL NOTES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "1. YOLO represents supervised object evidence.\n"
    )

    f.write(
        "2. VAE represents reconstruction/anomaly evidence.\n"
    )

    f.write(
        "3. Flow represents latent normality deviation.\n"
    )

    f.write(
        "4. Agreement is high only when multiple signals "
        "are simultaneously strong.\n"
    )

    f.write(
        "5. Disagreement is treated as uncertainty rather "
        "than automatically as anomaly.\n"
    )

    f.write(
        "6. Thresholds are based only on normal validation "
        "samples using P95/P99.\n"
    )

    f.write(
        "7. No threshold was selected using the final test set.\n"
    )

    f.write(
        "8. These thresholds are a conservative initial "
        "operational policy, not a claim of optimality.\n"
    )

    f.write(
        "9. The final test set should be evaluated only after "
        "the SAAD protocol is frozen.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)

for name, table in all_tables.items():

    normal = table[
        table["label"] == 0
    ]

    anthropogenic = table[
        table["label"] == 1
    ]

    print()
    print(
        name.upper()
    )

    print(
        f"Normal mean evidence       : "
        f"{safe_mean(normal['mean_evidence']):.6f}"
    )

    print(
        f"Anthropogenic mean evidence: "
        f"{safe_mean(anthropogenic['mean_evidence']):.6f}"
    )

    print(
        f"Normal mean agreement      : "
        f"{safe_mean(normal['agreement']):.6f}"
    )

    print(
        f"Anthropogenic mean agreement: "
        f"{safe_mean(anthropogenic['agreement']):.6f}"
    )


print()
print("=" * 70)
print("DECISION THRESHOLDS")
print("=" * 70)

print(
    threshold_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("PRIMARY CALIBRATION: PLATT")
print("=" * 70)

print()

print(
    primary[
        [
            "saad_decision"
        ]
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 70)
print("PRIMARY DECISION BY CLASS")
print("=" * 70)

primary_decision_class = (
    primary
    .groupby(
        [
            "label",
            "saad_decision",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    primary_decision_class
    .to_string()
)


print()
print("=" * 70)
print("PRIMARY DECISION BY DOMAIN")
print("=" * 70)

primary_domain = (
    primary
    .groupby(
        [
            "dataset",
            "saad_decision",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    primary_domain
    .to_string()
)


print()
print("=" * 70)
print("TOP REVIEW / ANOMALY CANDIDATES")
print("=" * 70)

display_columns = [
    "dataset",
    "label",
    "yolo_evidence",
    "vae_evidence",
    "flow_evidence",
    "mean_evidence",
    "agreement_strength",
    "disagreement",
    "strongest_evidence",
    "saad_decision",
    "decision_confidence",
]

available_display = [
    c
    for c in display_columns
    if c in candidate_table.columns
]

print(
    candidate_table[
        available_display
    ]
    .head(25)
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
    "  saad_primary_evidence_scores.csv"
)

print(
    "  evidence_scores_percentile.csv"
)

print(
    "  evidence_scores_platt.csv"
)

print(
    "  evidence_scores_isotonic.csv"
)

print(
    "  decision_thresholds.csv"
)

print(
    "  decision_summary.csv"
)

print(
    "  dataset_wise_evidence_summary.csv"
)

print(
    "  ranked_review_candidates.csv"
)

print(
    "  top_200_uncertain_candidates.csv"
)

print(
    "  saad_evidence_engine_summary.txt"
)