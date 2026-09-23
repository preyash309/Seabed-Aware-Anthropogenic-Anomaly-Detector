"""Exact live API ordering, summary, and response serialization."""


def serialize_response(
    candidates, survey_id, filename, image, width, height,
    unique_filename, API_BASE_URL, DEVICE,
):
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
            filename,

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
