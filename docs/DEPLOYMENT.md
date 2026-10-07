# Deployment runbook

The order matters: **bucket → keys → publish → check → run the image locally → host**. Each step ends with something you can
verify before moving on. The host is the last and the most replaceable step: the container only needs environment variables, one
port and outbound HTTPS to the bucket.

## 1. Create the bucket and two access keys (IDrive e2)

1. In the e2 dashboard create a **private bucket** and note its region.
2. Under **Enabled regions**, copy the **endpoint** for that region (it looks like `https://<id>.<region>.idrivee2-NN.com`). The
   endpoint is specific to your account and region, and keys only work against it.
3. Under **Access Keys**, create two keys, each limited to this one bucket:

   | Key | Permission | Lives on |
   |---|---|---|
   | publisher | read and write | the machine that trains and publishes. Never on the host. |
   | reader | read only | the container host, as a secret |

   A key's permission and bucket scope cannot be changed after it is created; make a new key instead.

## 2. Configure

```bash
cp .env.example .env              # fill in E2_ENDPOINT_URL, E2_BUCKET and the publisher key for now
set -a; source .env; set +a       # load it into this shell
```

## 3. Publish a model

```bash
make train evaluate export                       # produces models/best_model.onnx
python -m src.deploy.publish_model --dry-run     # loads the file as the service would; no network
python -m src.deploy.publish_model               # uploads the archive copy, then the current object
python -m src.deploy.publish_model --list        # versions in the bucket; * marks the current one
```

Done when `--list` shows your version with a `*`. The version looks like `20261007T201500Z-3fa9c21`: the export time and the git
commit (`-dirty` if the working tree had uncommitted changes).

## 4. Check the reader key

Put the **reader** key into your shell instead of the publisher key and run:

```bash
make check-bucket        # python -m src.serving.storage
```

It downloads the model exactly as the container will and prints the source object and the model metadata. Done when it prints
JSON; if it prints an error, see the troubleshooting table below. This is the one step I could not do for you, because it needs
your real bucket.

## 5. Run the image locally

```bash
docker build -t defect-api .
docker run --rm -p 8000:8000 --env-file .env defect-api     # .env now holds the reader key
```

Look for `Model <version> ready (resnet18, 6 classes, 224px input)` in the log, then:

```bash
curl localhost:8000/health
curl -F "file=@data/raw/NEU-DET/validation/images/crazing/crazing_241.jpg" localhost:8000/predict
```

## 6. Deploy to a host

Whatever the host, give it:

| Need | Value |
|---|---|
| Build | the `Dockerfile` in the repository root |
| Port | 8000 (the image listens on `$PORT` when the host sets it) |
| Environment | `E2_ENDPOINT_URL`, `E2_BUCKET`, `E2_ACCESS_KEY_ID`, `E2_SECRET_ACCESS_KEY` as secrets; optionally the others in `.env.example` |
| Health check | HTTP `GET /health`, start period of at least 60 s for the first model download |
| Memory | about 512 MB (ResNet-18 measured 233 MB resident); with less, set `ORT_THREADS=1` and `MAX_CONCURRENT_INFERENCES=1` |
| Network | outbound HTTPS to the e2 endpoint |

Free tiers change often, so check each provider's current limits. Notes for the usual candidates:

- **Hugging Face Spaces (Docker)**: the free CPU tier has the most memory of the free options, and secrets go under
  *Settings → Variables and secrets*. A Space needs a README with front matter, which this repository's README does not
  have (it would clutter GitHub), so create the Space from a copy:
  ```bash
  git clone https://huggingface.co/spaces/<you>/defect-api space && cd space
  cp -r ../Dockerfile ../requirements-serving.txt ../src .
  printf -- '---\ntitle: Defect detection API\nsdk: docker\napp_port: 8000\n---\n' > README.md
  git add -A && git commit -m "Deploy" && git push
  ```
  Spaces sleep when idle, so the first request after a pause waits for a container start and a model download.
- **Render**: New → Web Service → Docker; set *Health Check Path* to `/health` and add the environment variables. Free instances
  spin down when idle and have limited memory.
- **Koyeb**: create a service from the GitHub repository with the Dockerfile builder, port 8000, an HTTP health check on `/health`
  with a grace period of 60 s or more, and the `E2_*` values as secrets. A deployment that never turns healthy leaves the previous
  one serving.
- **Google Cloud Run**: `gcloud run deploy defect-api --source . --port 8000 --memory 1Gi` plus the environment variables (or
  Secret Manager). It scales to zero and needs a billing account on file.

Done when `GET /model` on the public URL shows the version you published.

## 7. Operating it

**Ship a new model**: `make train evaluate export`, then `python -m src.deploy.publish_model`, then restart or redeploy the service.
The image is not rebuilt. Confirm with `GET /model`.

**Roll back**: `python -m src.deploy.publish_model --list`, then `--promote <version>`, then restart the service.

**Rotate the reader key**: create a new reader key, update the secret on the host, redeploy, delete the old key.

**Logs**: one line per request (`POST /predict -> 200 in 22.9 ms request_id=...`) and one per prediction with the class,
confidence and `review=True/False`. The `request_id` is the `X-Request-ID` response header and is echoed in error bodies, so a
caller's report maps straight to a log line. Counting `review=True` over time is a cheap drift signal.

## 8. Troubleshooting

| You see | Meaning | Do this |
|---|---|---|
| `invalid configuration: E2_BUCKET is required; ...` | a variable is missing or malformed (all problems are listed at once) | set them; compare with `.env.example` |
| `HeadObject: access denied (403 ...)` | wrong key, the key cannot read this bucket, or the endpoint is not the one for the bucket's region | recheck keys and bucket scope; use the endpoint from *Enabled regions*; try `E2_REGION=<your region code>`; try `E2_ADDRESSING_STYLE=virtual` |
| `HeadObject: nothing at bucket ..., key ...` | wrong bucket or key name, or nothing is published | check `E2_BUCKET` and `E2_MODEL_KEY`; run `publish_model --list` with the publisher key |
| `gave up after 3 attempts: sha256 mismatch` | the object is damaged or was being replaced while downloading | restart; if it persists, publish again |
| warning `The object has no sha256 metadata` | the file was uploaded outside `publish_model` (for example the web console) | it still serves, with only its size checked; publish with the tool to record a checksum |
| `ONNX metadata is missing [...]` | the file was not produced by `export_onnx` | export again with `python -m src.models.export_onnx` |
| `onnxruntime could not load ...` | not a valid ONNX file, or it needs a newer onnxruntime than the one pinned in `requirements-serving.txt` | re-export; or raise the pin, rerun the tests, rebuild |
| `EndpointConnectionError` or a timeout | the endpoint is unreachable from the host | check the URL and the host's outbound network rules |
| first start is killed before it is healthy | the download took longer than the health check allows | raise the health check start period |
| container killed for memory | the host's limit is too low | more RAM, or `ORT_THREADS=1` and `MAX_CONCURRENT_INFERENCES=1` |
| `401 unauthorized` from `/predict` | `API_KEY` is set | send the `X-API-Key` header |

## 9. Security notes

- Only the reader key goes on the host. The publisher key stays on the machine that publishes.
- Secrets are never logged: the startup line prints the bucket and key names, not credentials.
- `API_KEY` protects `/predict` with a constant-time comparison but is a single shared secret; use a gateway for per-client keys or rate limits.
- `/docs` is served by default. Block it at your gateway if the API is public and you do not want the schema visible.
- TLS is the host's job: terminate HTTPS at the platform's edge.
