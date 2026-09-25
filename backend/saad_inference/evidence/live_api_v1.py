"""Frozen deployed evidence policy: saad-live-api-v1.

This is distinct from the offline Evidence Engine v3 research policy.
"""

from typing import Optional

import numpy as np

from config import Settings

_CALIBRATION = Settings.from_environment().calibration
YOLO_P5 = _CALIBRATION["normalization"]["yolo"]["p5"]
YOLO_P95 = _CALIBRATION["normalization"]["yolo"]["p95"]
VAE_P5 = _CALIBRATION["normalization"]["vae"]["p5"]
VAE_P95 = _CALIBRATION["normalization"]["vae"]["p95"]
FLOW_P5 = _CALIBRATION["normalization"]["flow"]["p5"]
FLOW_P95 = _CALIBRATION["normalization"]["flow"]["p95"]
YOLO_WEIGHT = _CALIBRATION["weights"]["yolo"]
VAE_WEIGHT = _CALIBRATION["weights"]["vae"]
FLOW_WEIGHT = _CALIBRATION["weights"]["flow"]


# ============================================================
# CALIBRATION HELPERS
# ============================================================

def percentile_score(
    value: Optional[float],
    p5: float,
    p95: float,
) -> float:

    """
    Convert a raw anomaly/evidence score into [0,1].

    Low values near validation-normal P5 -> ~0.
    High values near validation-normal P95 -> ~1.

    Values outside the range are clipped.
    """

    if value is None:
        return 0.0

    try:

        value = float(value)

    except Exception:

        return 0.0

    if not np.isfinite(value):

        return 0.0

    denominator = (
        p95 - p5
    )

    if denominator <= 0:

        return 0.0

    score = (
        value - p5
    ) / denominator

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# EVIDENCE ENGINE
# ============================================================

def build_evidence(
    yolo_confidence: float,
    vae_raw: Optional[float],
    flow_raw: Optional[float],
    tta_consistency: Optional[float],
):
    """
    Build the SAAD evidence profile.

    Important interpretation:

        YOLO:
            higher confidence = stronger anthropogenic evidence

        VAE:
            higher reconstruction error =
            stronger deviation from normal seabed

        RealNVP:
            higher NLL =
            stronger latent novelty

        TTA:
            higher consistency =
            more stable detector evidence

    TTA does NOT directly act as an anomaly probability.
    It primarily controls uncertainty.
    """

    # --------------------------------------------------------
    # Normalize individual evidence sources
    # --------------------------------------------------------

    yolo_evidence = percentile_score(
        yolo_confidence,
        YOLO_P5,
        YOLO_P95,
    )

    vae_evidence = percentile_score(
        vae_raw,
        VAE_P5,
        VAE_P95,
    )

    flow_evidence = percentile_score(
        flow_raw,
        FLOW_P5,
        FLOW_P95,
    )

    # --------------------------------------------------------
    # Base fused evidence
    # --------------------------------------------------------

    base_evidence = (
        YOLO_WEIGHT * yolo_evidence
        +
        VAE_WEIGHT * vae_evidence
        +
        FLOW_WEIGHT * flow_evidence
    )

    base_evidence = float(
        np.clip(
            base_evidence,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # TTA
    # --------------------------------------------------------

    if tta_consistency is None:

        tta = 0.5

    else:

        tta = float(
            np.clip(
                tta_consistency,
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------------
    # Evidence agreement
    #
    # Measures whether the independent evidence sources
    # broadly agree.
    # --------------------------------------------------------

    evidence_values = np.asarray(
        [
            yolo_evidence,
            vae_evidence,
            flow_evidence,
        ],
        dtype=np.float32,
    )

    evidence_std = float(
        np.std(
            evidence_values
        )
    )

    agreement = float(
        np.clip(
            1.0 - evidence_std * 1.5,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Cross-model disagreement
    # --------------------------------------------------------

    disagreement = float(
        np.clip(
            evidence_std * 1.5,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Corroboration
    #
    # Multiple evidence sources above the midline provide
    # stronger corroboration.
    # --------------------------------------------------------

    positive_sources = sum(
        value >= 0.50
        for value in evidence_values
    )

    if positive_sources >= 3:

        corroboration = 1.0

    elif positive_sources == 2:

        corroboration = 0.67

    elif positive_sources == 1:

        corroboration = 0.33

    else:

        corroboration = 0.0

    # --------------------------------------------------------
    # Priority
    #
    # Evidence strength is the main component.
    #
    # Corroboration provides a modest bonus.
    # TTA stability provides a modest bonus.
    # --------------------------------------------------------

    priority = (
        0.80 * base_evidence
        +
        0.10 * corroboration
        +
        0.10 * tta
    )

    priority = float(
        np.clip(
            priority,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Uncertainty
    #
    # High uncertainty means:
    #   - model disagreement
    #   - unstable TTA
    #   - weak evidence
    #
    # This is deliberately separate from priority.
    # --------------------------------------------------------

    uncertainty = (
        0.50 * disagreement
        +
        0.30 * (1.0 - tta)
        +
        0.20 * (1.0 - agreement)
    )

    uncertainty = float(
        np.clip(
            uncertainty,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Evidence profile
    # --------------------------------------------------------

    if (
        yolo_evidence >= 0.50
        and vae_evidence >= 0.50
        and flow_evidence >= 0.50
    ):

        evidence_profile = (
            "YOLO_VAE_FLOW_CORROBORATED"
        )

    elif (
        yolo_evidence >= 0.50
        and flow_evidence >= 0.50
    ):

        evidence_profile = (
            "YOLO_FLOW"
        )

    elif (
        yolo_evidence >= 0.50
        and vae_evidence >= 0.50
    ):

        evidence_profile = (
            "YOLO_VAE"
        )

    elif yolo_evidence >= 0.50:

        evidence_profile = (
            "DETECTOR_DOMINANT"
        )

    elif (
        vae_evidence >= 0.50
        or flow_evidence >= 0.50
    ):

        evidence_profile = (
            "NORMALITY_SIGNAL"
        )

    else:

        evidence_profile = (
            "WEAK_EVIDENCE"
        )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if uncertainty >= 0.70:

        recommendation = (
            "HIGH_PRIORITY_UNCERTAIN"
        )

        priority_level = "HIGH"

    elif priority >= 0.75:

        recommendation = (
            "HIGH_PRIORITY_REVIEW"
        )

        priority_level = "HIGH"

    elif priority >= 0.45:

        recommendation = (
            "REVIEW"
        )

        priority_level = "MEDIUM"

    elif (
        vae_evidence >= 0.70
        or flow_evidence >= 0.70
    ):

        recommendation = (
            "NOVELTY_REVIEW"
        )

        priority_level = "MEDIUM"

    elif uncertainty >= 0.40:

        recommendation = (
            "UNCERTAIN_REVIEW"
        )

        priority_level = "REVIEW"

    else:

        recommendation = (
            "LOW_PRIORITY_REVIEW"
        )

        priority_level = "LOW"

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "normalized": {

            "yolo":
                yolo_evidence,

            "vae":
                vae_evidence,

            "flow":
                flow_evidence,
        },

        "baseEvidence":
            base_evidence,

        "priority":
            priority,

        "uncertainty":
            uncertainty,

        "agreement":
            agreement,

        "disagreement":
            disagreement,

        "corroboration":
            corroboration,

        "ttaStability":
            tta,

        "evidenceProfile":
            evidence_profile,

        "priorityLevel":
            priority_level,

        "recommendation":
            recommendation,
    }
