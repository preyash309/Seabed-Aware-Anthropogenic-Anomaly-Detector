from pathlib import Path
from uuid import uuid4
from typing import Optional

import numpy as np
import torch

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image

from ultralytics import YOLO

from config import Settings
from saad_inference.evidence.live_api_v1 import build_evidence, percentile_score
from saad_inference.detector import decode_detection, predict as detect

from saad_inference.vae import (
    load_vae,
    extract_candidate_patch,
)

from saad_inference.realnvp import (
    load_realnvp,
    score_latent,
)

from saad_inference.tta import (
    compute_tta_consistency,
)


# ============================================================
# CONFIGURATION
# ============================================================

SETTINGS = Settings.from_environment()
SETTINGS.validate_assets()
SETTINGS.prepare_runtime_dirs()

UPLOAD_DIR = SETTINGS.upload_dir
YOLO_MODEL_PATH = SETTINGS.artifact_paths["yolo"]
DEVICE = SETTINGS.device
API_BASE_URL = SETTINGS.api_public_url


# ============================================================
# FROZEN EVIDENCE CALIBRATION
#
# These values were fitted from validation-normal statistics.
# They are NOT fitted on the held-out test set.
#
# YOLO:
#   P5  = 0.01089343
#   P95 = 0.17297355
#
# VAE:
#   P5  = 0.00106523
#   P95 = 0.03465850
#
# Flow:
#   P5  = 49.3122
#   P95 = 264.3907
# ============================================================

YOLO_P5 = SETTINGS.calibration["normalization"]["yolo"]["p5"]
YOLO_P95 = SETTINGS.calibration["normalization"]["yolo"]["p95"]

VAE_P5 = SETTINGS.calibration["normalization"]["vae"]["p5"]
VAE_P95 = SETTINGS.calibration["normalization"]["vae"]["p95"]

FLOW_P5 = SETTINGS.calibration["normalization"]["flow"]["p5"]
FLOW_P95 = SETTINGS.calibration["normalization"]["flow"]["p95"]


# ============================================================
# EVIDENCE WEIGHTS
#
# Primary evidence policy:
#
#   YOLO  = 0.50
#   VAE   = 0.20
#   Flow  = 0.30
#
# TTA is used primarily as a stability / uncertainty signal.
# ============================================================

YOLO_WEIGHT = SETTINGS.calibration["weights"]["yolo"]
VAE_WEIGHT = SETTINGS.calibration["weights"]["vae"]
FLOW_WEIGHT = SETTINGS.calibration["weights"]["flow"]


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SAAD Backend",
    version="0.5.0",
    description=(
        "SAAD — Seabed-Aware Anthropogenic "
        "Anomaly Detector backend"
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/uploads",
    StaticFiles(
        directory=str(UPLOAD_DIR)
    ),
    name="uploads",
)


# ============================================================
# MODEL LOADING
# ============================================================

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"YOLO model not found: "
        f"{YOLO_MODEL_PATH}"
    )


print("=" * 70)
print("SAAD BACKEND")
print("=" * 70)

print(
    "Device:",
    DEVICE,
)

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------

print(
    "Loading YOLO..."
)

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)

print(
    "YOLO loaded."
)


# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

print(
    "Loading VAE..."
)

vae_model = load_vae(
    checkpoint_path=SETTINGS.artifact_paths["vae"],
    device=DEVICE,
)

print(
    "VAE loaded."
)


# ------------------------------------------------------------
# RealNVP
# ------------------------------------------------------------

print(
    "Loading RealNVP..."
)

(
    flow_model,
    flow_latent_mean,
    flow_latent_std,
) = load_realnvp(
    flow_checkpoint=SETTINGS.artifact_paths["flow"],
    latent_mean_path=SETTINGS.artifact_paths["latent_mean"],
    latent_std_path=SETTINGS.artifact_paths["latent_std"],
    device=DEVICE,
)

print(
    "RealNVP loaded."
)


# ------------------------------------------------------------
# TTA
# ------------------------------------------------------------

print(
    "TTA: enabled"
)

print(
    "Evidence Engine: enabled"
)

print("=" * 70)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "operational",

        "service":
            "SAAD Backend",

        "version":
            "0.5.0",

        "modelSetId":
            SETTINGS.manifest["model_set_id"],

        "policyId":
            SETTINGS.calibration["policy_id"],

        "device":
            DEVICE,

        "cuda_available":
            torch.cuda.is_available(),

        "cuda_version":
            torch.version.cuda,

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "models": {

            "yolo":
                YOLO_MODEL_PATH.exists(),

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                True,
        },
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/api/model")
def model_info():

    return {

        "detector": {

            "name":
                "YOLO26s",

            "checkpoint":
                str(
                    YOLO_MODEL_PATH
                ),

            "loaded":
                True,
        },

        "vae": {

            "name":
                "ConvVAE-v1",

            "checkpoint": str(SETTINGS.artifact_paths["vae"]),

            "loaded":
                True,
        },

        "realnvp": {

            "name":
                "RealNVP",

            "checkpoint": str(SETTINGS.artifact_paths["flow"]),

            "latentMean": str(SETTINGS.artifact_paths["latent_mean"]),

            "latentStd": str(SETTINGS.artifact_paths["latent_std"]),

            "latentDim":
                128,

            "layers":
                8,

            "hidden":
                256,

            "loaded":
                True,
        },

        "tta": {

            "enabled":
                True,

            "perturbations": [
                "brightness",
                "contrast",
                "noise",
            ],

            "iouThreshold":
                0.30,
        },

        "evidenceEngine": {

            "enabled":
                True,

            "version":
                SETTINGS.calibration["policy_id"],

            "weights": {

                "yolo":
                    YOLO_WEIGHT,

                "vae":
                    VAE_WEIGHT,

                "flow":
                    FLOW_WEIGHT,
            },
        },

        "pipeline": {

            "yolo":
                True,

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                True,
        },

        "device":
            DEVICE,
    }


# ============================================================
# HELPERS
# ============================================================

def generate_survey_id():

    return (
        "SAAD-"
        +
        uuid4().hex[:8].upper()
    )


# ============================================================
# ANALYZE
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
):

    # ========================================================
    # VALIDATE UPLOAD
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )


    allowed_extensions = {

        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


    suffix = Path(
        file.filename
    ).suffix.lower()


    if suffix not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, BMP, "
                "TIFF or WEBP."
            ),
        )


    # ========================================================
    # SAVE UPLOAD
    # ========================================================

    survey_id = generate_survey_id()


    unique_filename = (
        uuid4().hex
        +
        suffix
    )


    output_path = (
        UPLOAD_DIR
        /
        unique_filename
    )


    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )


    output_path.write_bytes(
        contents
    )


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            output_path
        )

        image.load()

    except Exception as exc:

        output_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: {exc}"
            ),
        )


    width, height = image.size


    # ========================================================
    # ANALYSIS LOGGING
    # ========================================================

    print()
    print("-" * 70)

    print(
        f"ANALYSIS: {survey_id}"
    )

    print(
        f"Image   : {file.filename}"
    )

    print(
        f"Size    : {width} x {height}"
    )

    print(
        f"Device  : {DEVICE}"
    )

    print("-" * 70)


    # ========================================================
    # YOLO
    # ========================================================

    print(
        "Running YOLO..."
    )

    try:

        results = detect(yolo_model, output_path, DEVICE)

    except Exception as exc:

        print(
            "YOLO ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"YOLO inference failed: {exc}"
            ),
        )


    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = []

    base_tta_detections = []


    if results:

        result = results[0]


        if result.boxes is not None:

            boxes = result.boxes


            for index in range(
                len(boxes)
            ):

                # ------------------------------------------------
                # YOLO coordinates
                # ------------------------------------------------

                confidence, class_id, bbox, bbox_pixels = decode_detection(
                    boxes, index, width, height, base_tta_detections
                )


                # ====================================================
                # VAE + REALNVP
                # ====================================================

                print(
                    f"Candidate {index + 1:02d}: "
                    f"YOLO={confidence:.4f}"
                )

                vae_score = None
                flow_score = None


                try:

                    # ------------------------------------------------
                    # Extract candidate patch
                    # ------------------------------------------------

                    patch = (
                        extract_candidate_patch(
                            image,
                            bbox_pixels,
                        )
                    )


                    # ------------------------------------------------
                    # Convert patch to tensor
                    # ------------------------------------------------

                    patch_array = np.asarray(
                        patch,
                        dtype=np.float32,
                    )


                    patch_array /= 255.0


                    patch_tensor = (
                        torch.from_numpy(
                            patch_array
                        )
                    )


                    patch_tensor = (
                        patch_tensor
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )


                    patch_tensor = (
                        patch_tensor.to(
                            DEVICE,
                            non_blocking=True,
                        )
                    )


                    # ------------------------------------------------
                    # VAE encoder
                    #
                    # Deterministic mu is the exact representation
                    # used by the trained RealNVP.
                    # ------------------------------------------------

                    with torch.inference_mode():

                        mu, logvar = (
                            vae_model.encode(
                                patch_tensor
                            )
                        )


                        reconstruction = (
                            vae_model.decode(
                                mu
                            )
                        )


                        vae_score_tensor = (
                            torch.mean(
                                (
                                    reconstruction
                                    -
                                    patch_tensor
                                )
                                ** 2
                            )
                        )


                        vae_score = float(
                            vae_score_tensor
                            .detach()
                            .cpu()
                            .item()
                        )


                    # ------------------------------------------------
                    # RealNVP
                    #
                    # score_latent() performs the exact saved
                    # latent normalization internally.
                    # ------------------------------------------------

                    flow_score = (
                        score_latent(
                            flow_model,
                            flow_latent_mean,
                            flow_latent_std,
                            mu,
                        )
                    )


                    # ------------------------------------------------
                    # Safety checks
                    # ------------------------------------------------

                    if not np.isfinite(
                        vae_score
                    ):

                        raise RuntimeError(
                            "VAE score is not finite."
                        )


                    if not np.isfinite(
                        flow_score
                    ):

                        raise RuntimeError(
                            "RealNVP score is not finite."
                        )


                    print(
                        f"    VAE MSE  : "
                        f"{vae_score:.8f}"
                    )

                    print(
                        f"    Flow NLL : "
                        f"{flow_score:.4f}"
                    )


                except Exception as exc:

                    print(
                        "  ANOMALY SCORING ERROR:",
                        repr(exc),
                    )

                    vae_score = None
                    flow_score = None


                # ------------------------------------------------
                # Candidate object
                # ------------------------------------------------

                candidate = {

                    "id":
                        (
                            f"candidate-"
                            f"{index + 1:02d}"
                        ),

                    "bbox":
                        bbox,

                    "bbox_pixels":
                        bbox_pixels,

                    "yoloConfidence":
                        confidence,

                    "vaeScore":
                        vae_score,

                    "flowScore":
                        flow_score,

                    "ttaConsistency":
                        None,

                    "priority":
                        None,

                    "uncertainty":
                        None,

                    "priorityLevel":
                        "LOW",

                    "evidenceProfile":
                        "PENDING_EVIDENCE",

                    "recommendation":
                        "PENDING",

                    "reviewStatus":
                        "PENDING",

                    "classId":
                        class_id,

                    "className":
                        "anthropogenic",
                }


                candidates.append(
                    candidate
                )


    # ========================================================
    # TTA
    # ========================================================

    print()
    print(
        "Running TTA consistency..."
    )


    tta_scores = []


    if base_tta_detections:

        try:

            # ------------------------------------------------
            # RGB numpy image for perturbations
            # ------------------------------------------------

            tta_image = image.convert(
                "RGB"
            )


            image_array = np.asarray(
                tta_image
            )


            # ------------------------------------------------
            # TTA
            # ------------------------------------------------

            tta_scores = (
                compute_tta_consistency(

                    model=yolo_model,

                    image=image_array,

                    base_detections=
                        base_tta_detections,

                    device=DEVICE,

                    imgsz=640,
                )
            )


            print(
                "TTA completed."
            )


        except Exception as exc:

            print(
                "TTA ERROR:",
                repr(exc),
            )

            tta_scores = [
                None
                for _ in candidates
            ]

    else:

        print(
            "No YOLO candidates. "
            "Skipping TTA."
        )


    # ========================================================
    # ATTACH TTA SCORES
    # ========================================================

    for index, candidate in enumerate(
        candidates
    ):

        if index < len(
            tta_scores
        ):

            score = (
                tta_scores[index]
            )


            if score is not None:

                try:

                    score = float(
                        score
                    )


                    if np.isfinite(
                        score
                    ):

                        score = max(
                            0.0,
                            min(
                                1.0,
                                score,
                            ),
                        )

                    else:

                        score = None


                except Exception:

                    score = None


            candidate[
                "ttaConsistency"
            ] = score


        else:

            candidate[
                "ttaConsistency"
            ] = None


    # ========================================================
    # EVIDENCE ENGINE
    # ========================================================

    print()
    print(
        "Running evidence engine..."
    )


    for candidate in candidates:

        evidence = build_evidence(

            yolo_confidence=
                candidate[
                    "yoloConfidence"
                ],

            vae_raw=
                candidate[
                    "vaeScore"
                ],

            flow_raw=
                candidate[
                    "flowScore"
                ],

            tta_consistency=
                candidate[
                    "ttaConsistency"
                ],
        )


        # ----------------------------------------------------
        # Attach evidence
        # ----------------------------------------------------

        candidate[
            "priority"
        ] = evidence[
            "priority"
        ]


        candidate[
            "uncertainty"
        ] = evidence[
            "uncertainty"
        ]


        candidate[
            "priorityLevel"
        ] = evidence[
            "priorityLevel"
        ]


        candidate[
            "evidenceProfile"
        ] = evidence[
            "evidenceProfile"
        ]


        candidate[
            "recommendation"
        ] = evidence[
            "recommendation"
        ]


        # ----------------------------------------------------
        # Detailed evidence object
        # ----------------------------------------------------

        candidate[
            "evidence"
        ] = {

            "raw": {

                "yolo":
                    candidate[
                        "yoloConfidence"
                    ],

                "vae":
                    candidate[
                        "vaeScore"
                    ],

                "flow":
                    candidate[
                        "flowScore"
                    ],

                "tta":
                    candidate[
                        "ttaConsistency"
                    ],
            },

            "normalized":
                evidence[
                    "normalized"
                ],

            "baseEvidence":
                evidence[
                    "baseEvidence"
                ],

            "agreement":
                evidence[
                    "agreement"
                ],

            "disagreement":
                evidence[
                    "disagreement"
                ],

            "corroboration":
                evidence[
                    "corroboration"
                ],

            "ttaStability":
                evidence[
                    "ttaStability"
                ],
        }


        print(
            f"  {candidate['id']}: "
            f"priority="
            f"{candidate['priority']:.4f}, "
            f"uncertainty="
            f"{candidate['uncertainty']:.4f}, "
            f"profile="
            f"{candidate['evidenceProfile']}, "
            f"action="
            f"{candidate['recommendation']}"
        )


    print(
        "Evidence engine completed."
    )


    # ========================================================
    # SORT
    #
    # Rank by priority rather than raw YOLO confidence.
    # This is important because SAAD is an evidence-ranking
    # system, not simply a YOLO confidence list.
    # ========================================================

    candidates.sort(

        key=lambda candidate:
            candidate[
                "priority"
            ],

        reverse=True,
    )


    # --------------------------------------------------------
    # Re-number after sorting
    # --------------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        candidate["id"] = (
            f"candidate-"
            f"{index:02d}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    high_priority = sum(

        candidate[
            "priorityLevel"
        ]
        == "HIGH"

        for candidate in candidates
    )


    review = sum(

        candidate[
            "priorityLevel"
        ]
        in {
            "MEDIUM",
            "REVIEW",
        }

        for candidate in candidates
    )


    uncertain = sum(

        candidate[
            "recommendation"
        ]
        in {
            "HIGH_PRIORITY_UNCERTAIN",
            "UNCERTAIN_REVIEW",
        }

        for candidate in candidates
    )


    low_priority = sum(

        candidate[
            "priorityLevel"
        ]
        == "LOW"

        for candidate in candidates
    )


    print()
    print(
        f"Completed: "
        f"{len(candidates)} candidate(s)"
    )

    print(
        f"High      : {high_priority}"
    )

    print(
        f"Review    : {review}"
    )

    print(
        f"Uncertain : {uncertain}"
    )

    print(
        f"Low       : {low_priority}"
    )

    print(
        "-" * 70
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "surveyId":
            survey_id,

        "filename":
            file.filename,

        "image": {

            "width":
                width,

            "height":
                height,

            "format":
                image.format,

            "url":
                (
                    f"{API_BASE_URL}"
                    f"/uploads/"
                    f"{unique_filename}"
                ),
        },

        "processing": {

            "device":
                DEVICE,

            "detector":
                "YOLO26s",

            "imgsz":
                640,

            "confidenceThreshold":
                0.05,

            "tta":
                True,

            "evidenceEngine":
                True,
        },

        "summary": {

            "totalCandidates":
                len(candidates),

            "highPriority":
                high_priority,

            "review":
                review,

            "uncertain":
                uncertain,

            "lowPriority":
                low_priority,
        },

        "candidates":
            candidates,

        "pipeline": {

            "yolo":
                True,

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                True,
        },
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "SAAD Backend",

        "status":
            "operational",

        "version":
            "0.5.0",

        "pipeline":
            (
                "YOLO26s + VAE-v1 + "
                "RealNVP + TTA + "
                "Evidence Engine"
            ),
    }
