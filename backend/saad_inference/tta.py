from typing import List, Dict, Any

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

TTA_CONFIDENCE = 0.05

TTA_IOU_THRESHOLD = 0.30

# Weight given to spatial agreement.
IOU_WEIGHT = 0.70

# Weight given to confidence agreement.
CONFIDENCE_WEIGHT = 0.30


# ============================================================
# IOU
# ============================================================

def box_iou(
    box_a,
    box_b,
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1,
    )

    intersection_y1 = max(
        ay1,
        by1,
    )

    intersection_x2 = min(
        ax2,
        bx2,
    )

    intersection_y2 = min(
        ay2,
        by2,
    )

    intersection_width = max(
        0.0,
        intersection_x2
        - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2
        - intersection_y1,
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        /
        union_area
    )


# ============================================================
# IMAGE PERTURBATIONS
# ============================================================

def brightness_perturbation(
    image: np.ndarray,
) -> np.ndarray:

    image_float = image.astype(
        np.float32
    )

    # Mild brightness increase.
    output = (
        image_float
        *
        1.15
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


def contrast_perturbation(
    image: np.ndarray,
) -> np.ndarray:

    image_float = image.astype(
        np.float32
    )

    mean = (
        image_float.mean(
            axis=(0, 1),
            keepdims=True,
        )
    )

    # Mild contrast reduction.
    output = (
        (
            image_float
            -
            mean
        )
        *
        0.85
        +
        mean
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


def noise_perturbation(
    image: np.ndarray,
) -> np.ndarray:

    image_float = image.astype(
        np.float32
    )

    rng = np.random.default_rng(
        42
    )

    noise = rng.normal(
        loc=0.0,
        scale=5.0,
        size=image_float.shape,
    )

    output = (
        image_float
        +
        noise
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# EXTRACT YOLO DETECTIONS
# ============================================================

def extract_detections(
    result,
) -> List[Dict[str, Any]]:

    detections = []

    if result.boxes is None:

        return detections

    boxes = result.boxes

    for index in range(
        len(boxes)
    ):

        xyxy = (
            boxes.xyxy[index]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = float(
            boxes.conf[index]
            .detach()
            .cpu()
            .item()
        )

        class_id = int(
            boxes.cls[index]
            .detach()
            .cpu()
            .item()
        )

        detections.append(
            {
                "box": [
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                ],

                "confidence":
                    confidence,

                "classId":
                    class_id,
            }
        )

    return detections


# ============================================================
# MATCH BASE DETECTION AGAINST TTA DETECTIONS
# ============================================================

def match_detection(
    base_detection,
    tta_detections,
) -> Dict[str, float]:

    base_box = (
        base_detection["box"]
    )

    base_confidence = (
        base_detection["confidence"]
    )

    best_iou = 0.0

    best_confidence = 0.0

    for detection in tta_detections:

        # --------------------------------------------
        # Only compare the same class.
        # --------------------------------------------

        if (
            detection["classId"]
            !=
            base_detection["classId"]
        ):

            continue

        iou = box_iou(
            base_box,
            detection["box"],
        )

        if iou > best_iou:

            best_iou = iou

            best_confidence = (
                detection["confidence"]
            )

    # ------------------------------------------------
    # Confidence agreement.
    #
    # 1.0 = identical confidence
    # 0.0 = maximum disagreement.
    # ------------------------------------------------

    if best_iou >= TTA_IOU_THRESHOLD:

        confidence_difference = abs(
            base_confidence
            -
            best_confidence
        )

        confidence_agreement = max(
            0.0,
            1.0
            -
            confidence_difference,
        )

    else:

        # No spatially meaningful match.
        confidence_agreement = 0.0


    return {

        "iou":
            best_iou,

        "confidenceAgreement":
            confidence_agreement,
    }


# ============================================================
# TTA SCORE
# ============================================================

@torch.inference_mode()
def compute_tta_consistency(
    model,
    image: np.ndarray,
    base_detections: List[Dict[str, Any]],
    device: str,
    imgsz: int = 640,
) -> List[float]:

    if not base_detections:

        return []


    # --------------------------------------------------------
    # Generate photometric perturbations.
    # --------------------------------------------------------

    perturbed_images = [

        brightness_perturbation(
            image
        ),

        contrast_perturbation(
            image
        ),

        noise_perturbation(
            image
        ),
    ]


    # --------------------------------------------------------
    # Run YOLO on perturbations.
    # --------------------------------------------------------

    tta_detection_sets = []


    for index, perturbed in enumerate(
        perturbed_images,
        start=1,
    ):

        print(
            f"    TTA {index}/"
            f"{len(perturbed_images)}..."
        )


        results = model.predict(

            source=perturbed,

            imgsz=imgsz,

            conf=TTA_CONFIDENCE,

            device=device,

            verbose=False,
        )


        if results:

            detections = (
                extract_detections(
                    results[0]
                )
            )

        else:

            detections = []


        tta_detection_sets.append(
            detections
        )


    # --------------------------------------------------------
    # Calculate candidate-level consistency.
    # --------------------------------------------------------

    consistency_scores = []


    for base_detection in (
        base_detections
    ):

        perturbation_scores = []


        for tta_detections in (
            tta_detection_sets
        ):

            match = match_detection(

                base_detection,

                tta_detections,
            )


            iou_score = (
                match["iou"]
            )


            confidence_score = (
                match[
                    "confidenceAgreement"
                ]
            )


            consistency = (

                IOU_WEIGHT
                *
                iou_score

                +

                CONFIDENCE_WEIGHT
                *
                confidence_score
            )


            perturbation_scores.append(
                consistency
            )


        # ----------------------------------------------------
        # Mean stability over all perturbations.
        # ----------------------------------------------------

        if perturbation_scores:

            final_score = float(
                np.mean(
                    perturbation_scores
                )
            )

        else:

            final_score = 0.0


        consistency_scores.append(
            final_score
        )


    return consistency_scores