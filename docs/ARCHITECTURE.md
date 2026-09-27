# Architecture

This guide defines the implemented numerical pipeline and persistence boundaries. For installation use [Setup](SETUP.md); for endpoints and module ownership use [backend/README.md](../backend/README.md); for verification use [Evaluation](EVALUATION.md).

## Dependency flow

```text
web pages -> lib/api.ts -> main.py:app -> routes.py
                                      ├─ config.py -> model_manifest + calibration
                                      ├─ registry.py -> YOLO / VAE / flow loading
                                      ├─ service.py
                                      │   ├─ detector.py
                                      │   ├─ vae.py -> preprocessing.py
                                      │   ├─ realnvp.py
                                      │   ├─ tta.py
                                      │   ├─ evidence/live_api_v1.py
                                      │   └─ response.py
                                      ├─ store.py -> schemas.py
                                      └─ reporting.py
```

`main.py` preserves the FastAPI entry point and compatibility exports. Importing `routes.py` loads one model set per process; even health routes require assets at startup. Multiple processes load separate model sets. Legacy `vae_inference.py`, `realnvp_inference.py` and `tta_inference.py` remain compatibility shims.

Settings validate paths, the five tensor sizes/hashes and the raw live calibration hash before model loading. See [Configuration](../config/README.md). Checkpoint tests establish exact keys/shapes for 71 VAE and 56 flow state entries and 128 values each in latent mean/std. Runtime VAE loading retains the original behavior: missing keys fail and unexpected keys warn; the audited checkpoint has neither. Flow loading is strict. No checkpoint is rewritten.

## Detector and crop semantics

YOLO26s runs on the saved input path with `imgsz=640`, `conf=0.05`, the configured device and `verbose=False`. Detection boxes are clamped to image dimensions, with pixel `xyxy` and percentage `x,y,width,height` representations. No extra candidate suppression or replacement detector is introduced by the service.

For each candidate, crop extraction and service tensor preparation:

1. Converts the loaded image to grayscale with Pillow `convert("L")`.
2. Clamps box bounds, preserving at least a one-pixel extent.
3. Pads each side by 20% of the candidate width or height.
4. Centers a square with side equal to the larger padded extent and shifts it into the image at edges.
5. Rounds coordinates to integers before cropping.
6. Resizes to 256×256 with Pillow bilinear interpolation.
7. Converts to float32, divides by 255 and forms `[1,1,256,256]`.

These rules, including boundary shifts and rounding, are part of the frozen regression contract. Input is a raster image, not a raw acoustic or georeferenced sonar file.

## ConvVAE and latent flow

| ConvVAE stage | Channels / shape | Operation |
| --- | --- | --- |
| Input | 1×256×256 | Grayscale crop |
| Encoder | 1→32→64→128→256→512 | Five Conv2d blocks: kernel 4, stride 2, padding 1; BatchNorm and ReLU |
| Encoded feature | 512×8×8 = 32,768 | Flatten |
| Latent heads | 32,768→128 each | Mean and log variance; log variance clamped to [−10,10] |
| Decoder input | 128→32,768→512×8×8 | Linear and reshape |
| Decoder | 512→256→128→64→32→1 | Five ConvTranspose2d blocks: kernel 4, stride 2, padding 1; first four BatchNorm/ReLU, final Sigmoid |
| Output | 1×256×256 | Deterministic reconstruction from encoder mean |

Production uses `mu`, not a sampled latent. VAE score is the mean of squared reconstruction residuals over the crop.

RealNVP takes the 128-dimensional mean standardized as `(mu − latent_mean) / latent_std`; loaded std values are clamped to at least `1e-8`. Eight alternating affine coupling layers leave 64 coordinates fixed and transform the other 64. Each conditioner is a 64→256→256→128 MLP with ReLU hidden layers; its outputs provide scale and translation. Scale is `tanh(s) × 1.5`. The score is negative log likelihood under a standard normal base distribution, including the summed log determinant. Higher reconstruction error/NLL supports novelty relative to the learned normal seabed; neither establishes object identity.

## TTA consistency

The service repeats detection on RGB brightness ×1.15, contrast ×0.85 and additive Gaussian noise (σ=5, NumPy seed 42). The detector settings remain 640 / 0.05. For each original candidate, each variant selects the same-class box with greatest IoU.

Confidence agreement is `max(0, 1 − abs(base_confidence − matched_confidence))` when best IoU is at least 0.30, otherwise zero. Per-variant consistency is `0.70 × best_IoU + 0.30 × confidence_agreement`; final consistency is the mean of three variants, clipped to [0,1] by service orchestration. A subthreshold IoU still contributes its IoU term. No candidates means no TTA runs.

## Deployed policy: saad-live-api-v1

The committed calibration uses `validation_normals_only`, derived from the audited 59 normal validation images. Normalize each raw signal as `clip((raw − P5)/(P95 − P5), 0, 1)`. Missing or non-finite **raw model scores** normalize to zero.

| Signal | P5 | P95 | Base weight |
| --- | ---: | ---: | ---: |
| YOLO confidence | 0.01089343 | 0.17297355 | 0.50 |
| VAE MSE | 0.00106523 | 0.03465850 | 0.20 |
| Flow NLL | 49.3122 | 264.3907 | 0.30 |

Let `y,v,f` be normalized scores, `s` their float32 standard deviation, and `t` clipped TTA consistency (missing TTA uses 0.5).

```text
base          = clip(0.50*y + 0.20*v + 0.30*f, 0, 1)
disagreement  = clip(1.5*s, 0, 1)
agreement     = clip(1 - 1.5*s, 0, 1)
corroboration = {0: 0.00, 1: 0.33, 2: 0.67, 3: 1.00}[count(scores >= 0.50)]
priority      = clip(0.80*base + 0.10*corroboration + 0.10*t, 0, 1)
uncertainty   = clip(0.50*disagreement + 0.30*(1-t) + 0.20*(1-agreement), 0, 1)
```

These are review scales, not calibrated class probabilities. The policy itself only clips TTA; service orchestration separately converts non-finite TTA results to missing before policy evaluation.

Evidence profiles use inclusive ≥0.50 ties, in this order:

| Condition | Profile |
| --- | --- |
| YOLO, VAE and flow positive | YOLO_VAE_FLOW_CORROBORATED |
| YOLO and flow positive | YOLO_FLOW |
| YOLO and VAE positive | YOLO_VAE |
| YOLO positive | DETECTOR_DOMINANT |
| VAE or flow positive | NORMALITY_SIGNAL |
| None positive | WEAK_EVIDENCE |

Recommendations are first-match rules:

| Condition | Recommendation | Priority level |
| --- | --- | --- |
| uncertainty ≥0.70 | HIGH_PRIORITY_UNCERTAIN | HIGH |
| priority ≥0.75 | HIGH_PRIORITY_REVIEW | HIGH |
| priority ≥0.45 | REVIEW | MEDIUM |
| VAE ≥0.70 or flow ≥0.70 | NOVELTY_REVIEW | MEDIUM |
| uncertainty ≥0.40 | UNCERTAIN_REVIEW | REVIEW |
| Otherwise | LOW_PRIORITY_REVIEW | LOW |

Candidates are stably sorted by descending priority and assigned scan-local `candidate-01` etc. Summary high/review/low counts use priority levels; the uncertain count uses recommendations containing `UNCERTAIN` and can overlap those counts.

### Offline policy boundary

`saad-offline-evidence-v3` is archived in [build_saad_evidence_engine_v3.py](../research/original/Evidence/build_saad_evidence_engine_v3.py). It shares normal-reference ranges and primary weights, but uses different bonuses/penalties, uncertainty and profile/action rules; it does not use TTA. The final historical evaluation also explored `YOLO_FLOW_MEAN`. Neither is imported or selectable by the live API. Offline results cannot be labeled as live-policy results.

## Failure and serialization boundaries

Anomaly inference is one per-candidate VAE/flow block: failure of either leaves both scores absent. TTA failure uses the missing-consistency fallback. A response can therefore succeed with partial evidence; scan retrieval records `analysisStatus=partial`. A detector failure fails the request. [Backend errors](../backend/README.md#error-behavior) documents the HTTP boundary.

The compatibility response retains raw scores, normalized evidence, boxes, class, profile, recommendation, priority and uncertainty. Scan persistence validates the response separately. The original prediction's `reviewStatus` remains its inference-time value; saved human state is returned separately in `reviews`.

## Persistence and reports

SQLite stores four tables: `scans`, ordered `candidates`, current `reviews` and `review_events`. Foreign keys and WAL are enabled; transactions update the current review and event together. An unreviewed candidate is projected as PENDING, revision 0. Each action increments revision and records prior/current state. Accept/reject clears a previous correction. There is no reset-to-pending or deletion route.

The browser combines saved predictions and review state through `web/src/lib/adapt.ts`. A corrected box becomes the display box; the original model box remains separate. Reload/direct navigation fetches persisted state. Pending queue order uses saved model priority.

JSON reports include prediction/review data, events and model/policy metadata. CSV is a 22-column candidate export with raw signals, original/corrected percentage boxes and revision; it does not include the complete event trail or policy metadata. PDF is a text-only summary with metadata, evidence and boxes, not an image overlay or full event transcript. Model summary counts remain inference counts rather than accepted-object counts. See [Backend](../backend/README.md) and [Limitations](LIMITATIONS.md).
