"""Original analysis control flow, with frozen models supplied by the registry."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from config import Settings
from .registry import ModelRegistry
from .detector import decode_detection, predict as detect
from .vae import extract_candidate_patch
from .realnvp import score_latent
from .tta import compute_tta_consistency
from .evidence.live_api_v1 import build_evidence
from .response import serialize_response


def generate_survey_id():
    return "SAAD-" + uuid4().hex[:8].upper()


async def analyze(
    file: UploadFile,
    models: ModelRegistry,
    settings: Settings,
):
    yolo_model = models.yolo_model
    vae_model = models.vae_model
    flow_model = models.flow_model
    flow_latent_mean = models.flow_latent_mean
    flow_latent_std = models.flow_latent_std
    DEVICE = settings.device
    UPLOAD_DIR = settings.upload_dir

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


    return serialize_response(
        candidates, survey_id, file.filename, image,
        width, height, unique_filename, settings.api_public_url, DEVICE,
    )
