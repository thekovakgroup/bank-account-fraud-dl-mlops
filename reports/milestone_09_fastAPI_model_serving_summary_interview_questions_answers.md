# Milestone 09 — FastAPI Model Serving

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps  
**Milestone status:** Completed, merged into `main`, and synchronized locally  
**Repository:** `thekovakgroup/bank-account-fraud-dl-mlops`  
**Feature branch:** `milestone-09-model-api` (retained)  
**Feature commit:** `674e055`  
**Verified local `main` after merge:** `665c5c7`  
**Environment:** Windows, VS Code, PowerShell, Python 3.11.9, CPU-based PyTorch

> **Scope and limitations:** This is a **research-only fraud-risk scoring API**. It is **not approved for production banking decisions** and must not automatically reject applications. The final Month-7 holdout has already been consumed in an earlier milestone; Milestone 09 did **not** reuse it for tuning or model selection.

---

## 1. Executive Summary

In Milestone 09, I converted a previously trained, calibrated, and frozen PyTorch fraud-detection model into a locally served **FastAPI** application. The API accepts strictly validated bank-account application features, runs the existing fitted preprocessing and neural network, applies the previously fitted calibration, and returns a calibrated fraud-risk probability for each application. It **does not make automated approval, rejection, or investigation-selection decisions**.

I engineered supporting controls: startup/shutdown model lifecycle, a readiness probe separate from basic liveness, predictable client errors, sanitized inference-failure logging, a 100-application batch limit, and a 256 KiB HTTP body limit on `/predict`. I used dependency-injected fake inference engines for automated tests, separately performed a real-model smoke test, launched the local server with Uvicorn, verified Swagger/OpenAPI documentation, and merged the milestone into GitHub `main`.

**Verified result:** **40 API tests passed**, with **26 non-API tests deselected** in the final focused test command; the local API started and shut down gracefully; `/ready`, `/docs`, and `/openapi.json` returned HTTP 200 during the live-server check.

## 2. Goals and Deliverables

| Area | What was delivered | Why it matters |
|---|---|---|
| Serving | FastAPI application and `POST /predict` | Exposes reusable model inference through HTTP |
| Model integrity | Existing frozen model and preprocessing loaded, SHA-256 artifact checks | Prevents accidental substitution of artifacts |
| Input contract | Pydantic schema for 29 raw fields | Rejects malformed or unexpected inputs |
| Output contract | Validated risk probabilities and policy metadata | Keeps downstream integration explicit |
| Availability | `/health`, `/ready`, startup/shutdown lifecycle | Distinguishes a running process from a ready model |
| Failure handling | HTTP 422, 413, 503, and generic 500 where appropriate | Produces controlled operational behavior |
| Privacy | Sanitized error event logging | Avoids logging financial details or exception text |
| Resource bounds | 1–100 applications/request; 256 KiB `/predict` body cap | Reduces excessive processing of individual requests |
| Testing | Synthetic-engine automated tests plus real-engine smoke check | Supports validation without accessing the final holdout |
| Reproducibility | Pinned dependency records and startup instructions | Helps another developer understand the local setup |
| Git | Explicit staging, privacy filename screening, commit, PR, merge | Creates an auditable engineering checkpoint |

## 3. Architecture and Data Flow

```text
Client (synthetic application data)
             |
             v
  POST /predict (JSON body)
             |
             v
  Request-size middleware (<= 256 KiB)
             |
             v
  Pydantic validation (1–100 applications;
  exactly 29 expected input fields per application)
             |
             v
  Check app.state.model_ready and inference_engine
             |
             v
  pandas DataFrame of validated applications
             |
             v
  FraudInference.predict_proba(...)
       |                    |                    |
       v                    v                    v
  Saved fitted         Frozen PyTorch       Saved Platt
  preprocessor         MLP (CPU, eval)     calibrator
       |                    |                    |
       +--------------------+--------------------+
                            |
                            v
               Calibrated fraud probabilities
                            |
                            v
                 Pydantic response schema
                            |
                            v
            HTTP 200: scores + model/policy metadata
```

**Separately:** application startup initializes the inference engine once; shutdown clears it. `GET /health` checks liveness; `GET /ready` checks whether the engine was initialized and is present. The API does not implement the complete-cohort investigation selection algorithm.

### Frozen model and decision policy

- **Network architecture:** `55 → 64 → 32 → 1` (55 preprocessed inputs, hidden layers of 64 and 32, one output).
- **Serving inputs:** 29 raw features — **18 numeric, 5 categorical, and 6 binary** — transformed through the previously fitted preprocessor.
- **Inference:** frozen PyTorch checkpoint, CPU, evaluation mode, with `torch.inference_mode()`.
- **Calibration:** previously fitted **Platt scaling**; no refitting in the API.
- **Frozen threshold:** approximately **4.9296% (0.049296)** from earlier evaluation work.
- **Investigation capacity:** maximum **5% of a complete cohort**, subject to authorized human review.
- **Crucial distinction:** `/predict` returns **probabilities only**. It does **not** apply the threshold to grant/deny decisions or claim that a 5% cohort cap can be enforced on independent API batches.
- **Holdout protocol:** Month 7 was evaluated only after the relevant selections were frozen and cannot be reused to tune the model, threshold, or calibration.

### Input validation specifics

- Numeric fields use finite-number validation (no NaN or positive/negative infinity).
- Categorical fields require nonempty strict strings.
- Binary fields accept only the configured `0`/`1` values.
- The schema forbids extra fields; missing required features are rejected.
- Five numeric features support a **pre-established `-1` sentinel**; this is not permission to replace arbitrary missing values or change training-time preprocessing.
- Explicitly excluded non-serving columns include the target `fraud_bool`, `month`, and `device_fraud_count`.

## 4. API Contract

**Local server:** `http://127.0.0.1:8000`

| Method | Route | Normal purpose / response |
|---|---|---|
| `GET` | `/health` | `200 {"status":"alive"}` when the application responds; does not certify model readiness |
| `GET` | `/ready` | `200 {"status":"ready"}` when engine is marked ready and present; otherwise HTTP 503 |
| `POST` | `/predict` | HTTP 200 with calibrated probabilities and immutable model/policy metadata for validated input |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/openapi.json` | Machine-readable OpenAPI specification |

### Prediction request

`POST /predict` accepts a JSON object with an `applications` array containing **1–100** complete, valid 29-feature applications. Consult `app/schemas.py` or `/docs` for exact field names and types; do **not** invent substitute fields or send real customer data for demonstrations.

### Prediction response

The response schema includes:

- `model_version`: `sha256:<actual-frozen-checkpoint-hash>` — identifies the checkpoint without inventing a version number.
- `policy_status`: `FROZEN_PROVISIONAL_CANDIDATE`.
- `production_approved`: `false`.
- `decision_type`: `risk_score_only`.
- `predictions`: per-input records with zero-based `application_index` and finite `fraud_probability` between 0 and 1.

### Response and failure handling

| HTTP status | Meaning | Notes |
|---|---|---|
| 200 | Successful request | For prediction, returns risk scores; for operational endpoints, their normal status |
| 413 | Oversized `/predict` request body | Above **262,144 bytes**; checked against received bytes, not only Content-Length |
| 422 | Invalid request body or schema | Missing/extra fields, unsupported values, invalid JSON, invalid batch limits, etc. |
| 500 | Unexpected scoring error | Client sees generic `Fraud scoring failed.`; sanitized event is logged |
| 503 | Engine unavailable | `/ready` and `/predict` guard against unavailable inference |

**Boundary proof:** a 262,144-byte invalid JSON request passed the size gate and received HTTP **422**; a 262,145-byte request was rejected with HTTP **413**. Normal synthetic predictions continued working after middleware installation.

## 5. Operational and Security Engineering

### Model lifecycle

The application is built through `create_app(inference_factory=...)`, with a lifespan manager that initializes the model once, stores `app.state.inference_engine` and `app.state.model_ready`, and cleans up on shutdown. Synthetic factories make it testable without private artifacts. A simulated artifact-loading failure prevents successful startup and leaves the engine unavailable.

### Privacy-safe error logging

Unexpected inference failures produce only a standardized event and exception **class name**, for example:

```text
event=fraud_inference_failed exception_type=ValueError
```

The tested logger does not attach a traceback or log the raw exception message, private artifact path, request body, applicant features, or prediction value. Logging destinations, access control, and retention policies would still need deployment-specific configuration.

### Request limits

- FastAPI/Pydantic enforces **1–100 applications per call**.
- ASGI middleware enforces a **256 KiB** received-body limit on `POST /predict`.
- The byte limit is an initial local safeguard, **not** a benchmark-proven production limit. Public deployments still require authentication, authorization, transport security, rate limiting, and infrastructure protections.

### Model artifact handling

- Local private model/preprocessor/calibration artifacts are needed for the actual model to serve.
- The inference layer checks artifact SHA-256 hashes.
- Existing trusted `joblib` artifacts are loaded only from trusted local sources; PyTorch checkpoint restoration uses `weights_only=True`.
- No trained weights, raw datasets, Month-7 records, or MLflow databases were included among the 18 committed milestone files.
- A fresh GitHub clone alone cannot execute real inference until the required private artifacts are provisioned.

## 6. Project Files Added or Updated

### Application modules — six new files

| File | Role |
|---|---|
| `app/main.py` | FastAPI factory, routes, scoring and error handling |
| `app/model_lifecycle.py` | Model startup, readiness state, shutdown, fail-fast behavior |
| `app/schemas.py` | Request and individual-application Pydantic schemas |
| `app/response_schemas.py` | Strict response contract and research-only metadata |
| `app/safe_logging.py` | Sanitized inference-failure events |
| `app/request_limits.py` | 256 KiB ASGI body-size gate for `/predict` |

### Tests — nine new files

`tests/test_api_health.py`, `tests/test_api_lifecycle.py`, `tests/test_api_schemas.py`, `tests/test_api_responses.py`, `tests/test_api_predict.py`, `tests/test_api_readiness.py`, `tests/test_api_safe_logging.py`, `tests/test_api_request_limits.py`, and `tests/manual_real_api_smoke.py`.

The manual real-model smoke script is not part of pytest's normal `test_api_*` discovery pattern. The previously built `src/fraud_inference.py` module was reused, not retrained or replaced.

### Existing files updated

- `README.md`: actual milestone progression, holdout restrictions, architecture, API usage, research-only positioning.
- `requirements.txt`: direct FastAPI / Pydantic / HTTPX2 dependencies.
- `requirements-lock.txt`: synchronized local pinned environment (not independently proof-tested as a fresh-environment install).

## 7. Testing and Evidence

### Final focused API suite

```powershell
python -m pytest tests -k api -v
```

**Verified:** `40 passed, 26 deselected in 26.70s` on Windows / Python 3.11.9.

| Test area | Number of tests |
|---|---:|
| Health | 1 |
| Lifecycle, including failed initialization | 2 |
| Request schemas | 15 |
| Response schemas | 13 |
| Prediction endpoint | 4 |
| Readiness | 3 |
| Sanitized logging | 1 |
| Request body size | 1 |
| **Total** | **40** |

The suite primarily uses **synthetic inference engines**. A separate explicit real-model API smoke test passed earlier using a synthetic application (not the final holdout). A live Uvicorn session separately verified successful startup, `/ready` HTTP 200, Swagger `/docs` HTTP 200, `/openapi.json` HTTP 200, and graceful shutdown. The live browser checks did **not** constitute a real financial-customer prediction or production load test.

### Reproducibility and Git

- `python -m pip check` reported **no broken requirements**.
- The lock file was checked against the current installed packages.
- Git privacy screening flagged only intentional `.gitkeep` placeholders under otherwise private-data/model directories; the filename scan was not a comprehensive secret audit.
- Staged files were inspected explicitly rather than using `git add .`.
- Git whitespace checks passed after targeted cleanup.
- Feature commit: `674e055` — **18 files**, **2,356 insertions**, **17 deletions**.
- Pull request was merged; local `main` fast-forwarded from `e0cb9f0` to `665c5c7`.
- Final `git status`: on `main`, up to date with `origin/main`, **working tree clean**.
- Preserved earlier notebook stash was not applied or deleted; milestone branch was retained.

## 8. Local Runbook — Windows / VS Code

Start in the repository root with the project's existing virtual environment and trusted private artifacts already provisioned:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

When Uvicorn reports `Application startup complete.`, visit:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`
- `http://127.0.0.1:8000/docs`

Stop the server with **Ctrl+C**. For a focused repeatable regression check:

```powershell
python -m pytest tests -k api -v
```

**Do not** submit customer records, publish the unauthenticated local service to the internet, apply automatic adverse decisions, or rerun Month-7 holdout experiments.

---

## 9. Interview Questions and Sample Answers

Use the answers as a starting point, then explain the trade-offs in your own words. They are grounded in the implementation and checks completed during this milestone.

### A. Architecture and ML inference

**1. What was the main objective of Milestone 09?**  
To turn a frozen, calibrated PyTorch fraud model into a locally usable FastAPI risk-scoring service, with explicit contracts, operational safeguards, documentation, tests, and a merged Git milestone.

**2. Walk me through a prediction request from arrival to response.**  
The request goes through a 256 KiB body limit, Pydantic validates the 29-feature applications and batch size, the route checks model readiness, converts the records into a DataFrame, calls `FraudInference.predict_proba`, and returns validated calibrated probabilities with model/policy metadata.

**3. Why use FastAPI rather than calling the model from a notebook?**  
A notebook is useful for experimentation; FastAPI provides a reusable HTTP contract, automatic validation and documentation, and an application lifecycle that is easier to integrate with other systems.

**4. What is the difference between raw features and transformed features in this project?**  
The API expects **29 raw application fields**, while the frozen preprocessor transforms them into **55 model inputs**. Clients should not attempt to submit the transformed representation directly.

**5. Why must preprocessing be loaded with the model?**  
The model was trained on features produced by a specific fitted transformer. Re-fitting or changing feature ordering at inference time could produce inconsistent or invalid predictions.

**6. Why put the PyTorch model into evaluation mode and use `torch.inference_mode()`?**  
Evaluation mode uses inference behavior for layers such as dropout/batch normalization if present. `inference_mode()` prevents autograd tracking and reduces overhead; the API does not need gradients or training.

**7. What is Platt calibration and why does the API use it?**  
Platt scaling is a fitted mapping from model scores to estimated probabilities, commonly using a logistic function. Our API loads the frozen calibrator so it reports the previously calibrated risk probabilities rather than uncalibrated network scores.

**8. Why didn't you retrain the model while building the API?**  
Serving is an engineering milestone. Retraining would change the selected model, invalidate frozen-artifact identity, and risk turning the final holdout into repeated model-selection feedback.

**9. How is the model version identified?**  
The response includes the actual frozen checkpoint's SHA-256 fingerprint as a `sha256:`-prefixed model identifier. That ties the response to model bytes, although full lineage would also record the preprocessor and calibrator artifact versions.

**10. Why is `FraudInference` separated from the API routes?**  
It separates ML inference from HTTP concerns, so the same inference logic can be used independently and the API can be tested with a fake engine without private files.

### B. Fraud policy and responsible AI

**11. Why is PR-AUC especially relevant to fraud detection?**  
Fraud is rare, so accuracy may be misleading. PR-AUC focuses on precision–recall trade-offs for the positive class and is useful for ranking under strong class imbalance. Operational decisions still require capacity and cost analysis.

**12. What does a 4.9296% threshold mean?**  
It is the frozen operational probability threshold of approximately 0.049296, established earlier. It does **not** mean a 4.9296% false-positive rate, and the API does not itself apply that threshold to make decisions.

**13. Why can't you enforce a 5% investigation cap on each independent API batch?**  
The cap applies to a **complete cohort** and depends on cohort-wide prioritization. Independent batches may contain different score distributions, so allocating 5% inside each batch does not reproduce a global cohort ranking or policy.

**14. Does a high fraud probability mean an application should be rejected?**  
No. The model is an investigation-support signal, not conclusive proof of fraud. The API explicitly returns `production_approved: false` and `decision_type: risk_score_only`; any investigation requires authorized human review.

**15. How did you avoid data leakage or test-set overfitting in this milestone?**  
Milestone 09 reused previously frozen preprocessing, model, calibration, and policy. API testing used synthetic applications and fake engines; the consumed Month-7 final holdout was not reused for tuning or model selection.

### C. API engineering and validation

**16. Why Pydantic?**  
It defines and validates the input/output contracts. It rejects missing or unexpected fields, invalid types, nonfinite numeric values, and invalid probability ranges before malformed data propagates.

**17. Why reject extra fields such as `fraud_bool`?**  
`fraud_bool` is the label, not an application-time input. Rejecting extra fields avoids ambiguous requests and reduces the risk of clients providing unavailable or leakage-prone fields.

**18. What is the difference between HTTP 413 and HTTP 422 here?**  
413 means the request body exceeded the **byte-size limit** before normal processing. 422 means the request reached application validation but its JSON/schema was unacceptable.

**19. Why did you test 262,144 and 262,145 bytes?**  
Boundary-value testing catches off-by-one errors. The configured maximum is inclusive: 262,144 bytes can reach validation; one byte beyond must receive 413.

**20. What do you do if the model returns a different number of probabilities than input rows?**  
The API treats it as a failed inference contract, raises an error internally, and responds with a generic HTTP 500 rather than returning mismatched predictions.

**21. Why include `application_index` in each prediction?**  
It preserves correspondence to the original request order, especially for batches of multiple applications.

**22. What does Swagger UI give you?**  
Automatically generated interactive API documentation using the OpenAPI schema. It makes endpoints and data contracts discoverable, but viewing Swagger does not establish production readiness.

### D. Reliability, security, and testing

**23. What's the difference between `/health` and `/ready`?**  
`/health` indicates the API process is responding. `/ready` indicates the inference engine is initialized and available. The service can be alive but not ready, in which case readiness returns 503.

**24. Why initialize the model at application startup?**  
Loading once avoids repeated model-load latency on every prediction and allows startup to fail promptly if essential artifacts are unavailable.

**25. What happens if a model artifact is missing during startup?**  
The lifespan initialization fails, preventing the app from successfully entering its serving state. Tests confirmed `model_ready` remains false and `inference_engine` remains absent for a simulated failure.

**26. How did you avoid exposing sensitive data in logs?**  
The logger records a fixed failure event and exception class only. We tested that synthetic financial markers, a mock private path, exception text, and attached tracebacks did not appear in its log record.

**27. Why not return a raw exception to the API client?**  
Raw exception messages can leak paths, internal structure, or data. The client receives a controlled generic 500 response while operators can identify the exception category through sanitized logging.

**28. Why did you use dependency injection for the inference engine?**  
The application factory accepts an engine factory, so unit/integration tests can inject a synthetic engine rather than loading private model artifacts. That makes API behavior faster to reproduce and isolates engineering failures from ML behavior.

**29. What is the difference between a unit, integration, and smoke test in your project?**  
A unit test checks a small rule (such as validation or logging); an integration test checks components working together (such as `/predict` calling the logger); a smoke test checks that a minimally realistic service/model path starts and responds. The final 40-test suite is not a replacement for security, deployment, or load testing.

**30. Are 40 passing tests enough to declare the service production-ready?**  
No. These tests cover specified functional behaviors, not full security, uptime, throughput, privacy governance, fairness, or deployment controls. The current research system has no public-production authorization.

### E. Reproducibility and Git/MLOps process

**31. What is the difference between `requirements.txt` and `requirements-lock.txt` here?**  
The first lists the project’s selected direct dependencies; the second captures a pinned local environment, including transitive packages. Matching installed packages to a lock file does not by itself prove that a fresh machine can reproduce the installation.

**32. Why use `git diff --cached --check`?**  
It checks **staged** changes for whitespace and patch-format errors before committing. We corrected trailing whitespace, restaged only the affected files, and reran it successfully.

**33. Why avoid `git add .` in this repository?**  
The working directory includes private datasets, model artifacts, and MLflow files. Explicit staging reduces the chance of accidentally adding sensitive or unrelated files. Gitignore and content review are still important.

**34. Why use a milestone branch and pull request?**  
It separates development from stable `main`, provides an auditable review boundary, and allows tests and documentation to be completed before the change is merged.

**35. What did `git pull --ff-only origin main` protect against?**  
It updated local `main` only if Git could fast-forward without creating an implicit merge commit. The actual synchronization succeeded from `e0cb9f0` to `665c5c7`.

**36. How would you improve this API before a real deployment?**  
Add secure artifact distribution, authentication/authorization, HTTPS, rate limiting, infrastructure limits, safe operational telemetry, dependency/build reproducibility from a clean environment, container/CI checks, deployment monitoring, model governance and formal risk approval. A cohort-wide investigation workflow would need its own controlled process.

---

## 10. Interview Practice — Questions to Answer Without Notes

1. Explain why an API could be **alive** but **not ready**, and give its expected HTTP status.
2. Describe the difference between a **probability**, a **threshold**, and an **investigation capacity budget**.
3. Explain why serving 29 raw features to a model with 55 inputs is correct.
4. Justify using a frozen preprocessor and frozen Platt calibrator.
5. Draw the `/predict` flow and identify where 413, 422, 503, and 500 errors arise.
6. Describe how dependency injection allowed tests without access to private artifacts.
7. Explain why the Month-7 holdout cannot be used again for tuning after final evaluation.
8. Name three controls already implemented and three missing before production deployment.
9. Defend the choice to return risk scores without automatic account rejection.
10. Describe the branch → review → commit → push → PR → merge → clean `main` workflow.

## 11. Sample Interview Pitch (60–90 Seconds)

> “For Milestone 09 of my bank account fraud detection capstone, I built a research-only FastAPI service around a previously frozen and Platt-calibrated PyTorch MLP. The model takes 29 raw application features, applies its saved preprocessor to produce the 55 inputs the network expects, and returns calibrated fraud probabilities. I kept model selection and the final Month-7 holdout separate from serving, and I deliberately avoided automating banking decisions or enforcing a 5% investigation cap on incomplete request batches.
>
> On the engineering side, I implemented strict Pydantic contracts, liveness and readiness endpoints, a startup/shutdown lifecycle, fail-fast loading, sanitized error logging, and a 256 KiB HTTP request-body guard. I used synthetic engines for repeatable API tests, separately checked the real model, verified the live service with Uvicorn and Swagger, and got 40 API tests passing. I then documented the usage, staged only intended files, completed a pull request, and merged into clean `main`. The next step is containerization, CI/CD, and registry/deployment engineering—not claiming this is production-approved.”

## 12. Lessons Learned and Next Milestone

**Key lessons:** a working model is not yet a serving system; the API contract must be separate from the training pipeline; final-holdout discipline carries into serving; liveness differs from readiness; bounded input and sanitized logs are practical MLOps safeguards; and focused automated tests plus disciplined Git workflow create verifiable progress.

**Milestone 10:** Docker/container reproducibility, GitHub Actions CI/CD, and MLflow Model Registry work. Preserve the frozen artifacts and evaluation policy. Confirm Docker availability and registry/artifact design before implementing the next engineering steps.

---

**Milestone 09 outcome:** **Completed and merged**. The local `main` branch was verified synchronized and clean. The system remains **research-only**; production approval and later deployment milestones are separate work.
