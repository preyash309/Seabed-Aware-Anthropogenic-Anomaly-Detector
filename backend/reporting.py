"""Server-backed exports from immutable predictions and durable review state."""

from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def report_data(detail: dict, events: list[dict], model_set_id: str, policy_id: str) -> dict:
    analysis = detail["analysis"]
    return {
        "reportType": "SAAD Sonar Anomaly Analysis",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scanId": detail["scanId"],
        "createdAt": detail["createdAt"],
        "modelSetId": model_set_id,
        "policyId": policy_id,
        "analysisStatus": detail["analysisStatus"],
        "filename": analysis["filename"],
        "image": analysis["image"],
        "summary": analysis["summary"],
        "candidates": [
            {"prediction": prediction, "review": detail["reviews"][prediction["id"]]}
            for prediction in analysis["candidates"]
        ],
        "reviewEvents": events,
    }


def _csv_safe(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def report_csv(data: dict) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "scan_id", "candidate_id", "class_id", "priority", "uncertainty",
        "priority_level", "evidence_profile", "recommendation", "yolo_confidence",
        "vae_mse", "flow_nll", "tta_consistency", "model_x", "model_y",
        "model_width", "model_height", "review_status", "corrected_x",
        "corrected_y", "corrected_width", "corrected_height", "review_revision",
    ])
    for item in data["candidates"]:
        prediction, review = item["prediction"], item["review"]
        box, corrected = prediction["bbox"], review["correctedBBox"] or {}
        writer.writerow([_csv_safe(value) for value in [
            data["scanId"], prediction["id"], prediction["classId"],
            prediction["priority"], prediction["uncertainty"],
            prediction["priorityLevel"], prediction["evidenceProfile"],
            prediction["recommendation"], prediction["yoloConfidence"],
            prediction["vaeScore"], prediction["flowScore"],
            prediction["ttaConsistency"], box["x"], box["y"],
            box["width"], box["height"], review["status"],
            corrected.get("x"), corrected.get("y"), corrected.get("width"),
            corrected.get("height"), review["revision"],
        ]])
    return output.getvalue()


def report_pdf(data: dict) -> bytes:
    output = BytesIO()
    document = canvas.Canvas(output, pagesize=A4)
    document.setTitle(f"SAAD report {data['scanId']}")
    width, height = A4
    y = height - 48

    def line(message: str, size: int = 9):
        nonlocal y
        if y < 54:
            document.showPage()
            y = height - 48
        document.setFont("Helvetica", size)
        document.drawString(45, y, message[:115])
        y -= size + 8

    line("SAAD Sonar Anomaly Analysis", 16)
    line(f"Scan: {data['scanId']}  Model: {data['modelSetId']}")
    line(f"Policy: {data['policyId']}  Status: {data['analysisStatus']}")
    line(f"Source: {data['filename']}")
    line(f"Generated: {data['generatedAt']}")
    line(f"Candidates: {len(data['candidates'])}")
    y -= 8
    for item in data["candidates"]:
        p, r = item["prediction"], item["review"]
        line(f"{p['id']}  Priority {p['priority']:.4f}  {p['priorityLevel']}  Review {r['status']}")
        line(f"  YOLO {p['yoloConfidence']:.6f}  VAE {p['vaeScore']}  Flow {p['flowScore']}  TTA {p['ttaConsistency']}")
        line(f"  Model box (%) {p['bbox']}")
        if r["correctedBBox"]:
            line(f"  Corrected box (%) {r['correctedBBox']}")
    document.save()
    return output.getvalue()
