# Example Predictions

The model classifies steel surface defects in real time. Below are predictions on validation images.

## Example 1: Crazing Defect
Correctly identified crazing with high confidence.

```json
{
  "predicted_class": "crazing",
  "confidence": 0.9709,
  "is_defective": true,
  "needs_review": false,
  "probabilities": {
    "crazing": 0.9709,
    "inclusion": 0.006,
    "patches": 0.0064,
    "pitted_surface": 0.0044,
    "rolled-in_scale": 0.0089,
    "scratches": 0.0034
  },
  "model_version": "20261007T222015Z-9d98af8-dirty",
  "inference_ms": 26.77
}
```

**Test:** 
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/NEU-DET/validation/images/crazing/crazing_241.jpg"
```

## Test Against the API

Start the server:
```bash
MODEL_PATH=models/best_model.onnx python -m src.serving.app
```

Then run predictions on any JPEG, PNG, BMP, or WebP image:
```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@your_image.jpg"
```

Interactive docs at: **http://127.0.0.1:8000/docs**

## Response Shape

All responses follow this schema:

| Field | Type | Notes |
|-------|------|-------|
| `predicted_class` | string | One of: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| `confidence` | float | Probability [0, 1] of the predicted class |
| `is_defective` | boolean | Always `true` (model has no "normal" class) |
| `needs_review` | boolean | `true` if confidence < 0.6 (heuristic for manual inspection) |
| `probabilities` | object | Softmax scores for all 6 classes |
| `model_version` | string | Identifier from the ONNX file (git commit + timestamp) |
| `inference_ms` | float | Time to run the model (excludes image decode) |

## Errors

All errors return this JSON shape with a stable `code`:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "the uploaded file could not be decoded as a valid image",
    "request_id": "abc123def456"
  }
}
```

Common codes:
- `invalid_image` — not a valid JPEG/PNG/BMP/WebP
- `empty_file` — upload is 0 bytes
- `file_too_large` — exceeds 10 MB limit
- `validation_error` — multipart field `file` is missing
- `unauthorized` — if `X-API-Key` header is required but missing

Use the `request_id` to trace errors in the server logs.
