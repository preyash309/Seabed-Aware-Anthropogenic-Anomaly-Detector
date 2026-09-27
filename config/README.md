# Configuration and frozen assets

This directory contains [model_manifest.json](model_manifest.json) and [live_api_v1.json](live_api_v1.json). Runtime settings are implemented in [backend/config.py](../backend/config.py); copy [.env.example](../.env.example) to ignored `.env`. Process environment values take precedence over dotenv values. Relative paths resolve from the repository root.

## Asset identities and placement

Model set: `saad-original-frozen-2026-09-23`. Deployed policy: `saad-live-api-v1`; research policy identifier: `saad-offline-evidence-v3`.

Supply authorized original tensors under an external `SAAD_ARTIFACT_DIR`:

```text
<external asset root>/
  yolo26s_generic_baseline/weights/best.pt
  vae_normal_seabed/checkpoints/best.pt
  latent_normalizing_flow/best_flow.pt
  latent_normalizing_flow/latent_mean.pt
  latent_normalizing_flow/latent_std.pt
```

| Manifest key | Bytes |
| --- | ---: |
| yolo | 20,303,429 |
| vae | 218,365,942 |
| flow | 3,714,807 |
| latent_mean | 2,117 |
| latent_std | 2,110 |

The manifest is the authoritative SHA-256 list. All five asset sizes/hashes and the calibration's raw SHA-256 are checked before model loading. An artifact root or all five individual overrides are required. Similarly named experimental weights are not interchangeable. No default points runtime loading at the original project.

The live calibration is committed here; external research calibration/result files remain outside Git. A calibration override must match the pinned hash, policy identity and validated normalization/weights. The current manifest requires exactly five artifacts and the live policy. It is not a switch for deploying offline v3.

## Runtime environment reference

| Variable | Default / requirement | Meaning |
| --- | --- | --- |
| `SAAD_ARTIFACT_DIR` | Required unless all five overrides are supplied | External root for manifest-relative tensor paths. |
| `SAAD_YOLO_WEIGHTS` | Manifest path under root | Individual YOLO override. |
| `SAAD_VAE_WEIGHTS` | Manifest path under root | Individual ConvVAE override. |
| `SAAD_FLOW_WEIGHTS` | Manifest path under root | Individual RealNVP override. |
| `SAAD_LATENT_MEAN` | Manifest path under root | Individual latent-mean override. |
| `SAAD_LATENT_STD` | Manifest path under root | Individual latent-std override. |
| `SAAD_MODEL_MANIFEST` | `config/model_manifest.json` | Manifest file; schema/policy validated. |
| `SAAD_CALIBRATION_FILE` | `config/live_api_v1.json` | Hash-pinned live calibration file. |
| `SAAD_DATASET_DIR` | Unset | Optional read-only prepared dataset; if set, must exist. Not needed by runtime inference. |
| `SAAD_UPLOAD_DIR` | `var/uploads` | Upload directory inside the clean checkout. |
| `SAAD_OUTPUT_DIR` | `var/results` | Output directory inside the clean checkout. |
| `SAAD_DATABASE_PATH` | `var/saad.sqlite3` | SQLite file inside the clean checkout. |
| `SAAD_MAX_UPLOAD_BYTES` | `26214400` | Positive byte limit (25 MiB). |
| `SAAD_MAX_IMAGE_PIXELS` | `50000000` | Positive width × height limit. |
| `SAAD_API_PUBLIC_URL` | `http://127.0.0.1:8000` | Base used in returned image URLs. |
| `SAAD_CORS_ORIGINS` | localhost and 127.0.0.1 on port 5173 | Comma-separated HTTP(S) browser origins; CORS is not authentication. |
| `SAAD_DEVICE` | `auto` | CUDA device 0 when available, otherwise CPU; explicit `cpu` or available `cuda:N`. CPU parity is unverified. |

Runtime directories and database paths are confined to the resolved clean-repository root. External weights and optional datasets are read inputs, not output destinations. Startup prepares runtime directories; reports are serialized on demand, not automatically written to `SAAD_OUTPUT_DIR`.

## Verification and browser settings

These settings belong to verification/tools or the browser rather than the runtime Settings object:

| Variable | Use |
| --- | --- |
| `SAAD_GOLDEN_IMAGE_DIR` | Optional authorized mirror of the three fixture image basenames for preflight/golden tests. Hashes must still match. |
| `YOLO_AUTOINSTALL=False` | Prevents Ultralytics dependency auto-install during documented runs. |
| `YOLO_CONFIG_DIR` | Set to ignored `var/ultralytics` to confine Ultralytics state to the clean checkout. |
| `VITE_API_BASE_URL` | Browser build-time base URL; default `http://127.0.0.1:8000`. Set in ignored `web/.env.local`; rebuild after changes. |

Builder-only settings are documented in [ml/README.md](../ml/README.md). They do not configure production inference.

## Calibration bytes and LF

The calibration SHA-256 is `78677e9b6270615a389735f486decbe4ab623d5ae447a96be9e82c8517c14435`. [.gitattributes](../.gitattributes) sets `config/live_api_v1.json text eol=lf`. Windows CRLF conversion changes the raw hash even when JSON values are identical.

Do not resave the file with altered line endings, reformat it or update the manifest hash to bypass validation. Restore the committed LF bytes if checkout/editor conversion occurred. Startup checks calibration policy and numeric structure as well as raw identity. [Evaluation](../docs/EVALUATION.md) records the prior CRLF failure and resolution.

Use [Setup](../docs/SETUP.md) for launch commands and [research verification](../research/README.md) for asset/image preflight. Never commit model files, datasets, local environment files or runtime state.
