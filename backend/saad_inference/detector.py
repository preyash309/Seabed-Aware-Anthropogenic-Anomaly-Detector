"""Frozen YOLO26s prediction and original candidate box decoding."""


def predict(model, output_path, device):
    return model.predict(
        source=str(output_path),
        imgsz=640,
        conf=0.05,
        device=device,
        verbose=False,
    )


def decode_detection(boxes, index, width, height, base_tta_detections):
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


    x1, y1, x2, y2 = map(
        float,
        xyxy,
    )


    # ------------------------------------------------
    # Clamp
    # ------------------------------------------------

    x1 = max(
        0.0,
        min(x1, width),
    )

    y1 = max(
        0.0,
        min(y1, height),
    )

    x2 = max(
        0.0,
        min(x2, width),
    )

    y2 = max(
        0.0,
        min(y2, height),
    )


    box_width = max(
        0.0,
        x2 - x1,
    )

    box_height = max(
        0.0,
        y2 - y1,
    )


    # ------------------------------------------------
    # Preserve raw detection for TTA
    # ------------------------------------------------

    base_tta_detections.append(
        {
            "box": [
                x1,
                y1,
                x2,
                y2,
            ],

            "confidence":
                confidence,

            "classId":
                class_id,
        }
    )


    # ------------------------------------------------
    # Percentage bbox
    # ------------------------------------------------

    bbox = {

        "x":
            (
                x1
                /
                width
            )
            *
            100.0,

        "y":
            (
                y1
                /
                height
            )
            *
            100.0,

        "width":
            (
                box_width
                /
                width
            )
            *
            100.0,

        "height":
            (
                box_height
                /
                height
            )
            *
            100.0,
    }


    bbox_pixels = {

        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,
    }

    return confidence, class_id, bbox, bbox_pixels
