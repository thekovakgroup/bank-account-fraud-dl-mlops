# Milestone 10 — Docker, CI/CD, and MLflow Model Registry

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps  
**Milestone status:** Complete — merged into `main`  
**Verified through:** October 20, 2026  
**Repository:** [thekovakgroup/bank-account-fraud-dl-mlops](https://github.com/thekovakgroup/bank-account-fraud-dl-mlops)  
**Pull request:** [#12 — Docker CI and frozen MLflow Model Registry](https://github.com/thekovakgroup/bank-account-fraud-dl-mlops/pull/12)  
**Final merge commit:** `89eff35`  
**Model status:** `FROZEN_PROVISIONAL_CANDIDATE`; **not approved for production**

## 1. Executive summary

Milestone 10 introduced repeatable **container-build validation**, a **GitHub Actions CI workflow**, and a **local MLflow Model Registry** for the previously frozen weighted PyTorch multilayer perceptron (MLP). It did **not** create or tune a new model. Instead, the milestone focused on preserving the exact trained model, fitted preprocessing, and Platt probability calibration in a verifiable MLflow inference package.

The saved MLflow model was reloaded from the local export and then from **registered model version 1**. On synthetic applications, both returned calibrated probabilities matching the frozen inference reference within the test tolerance. The full local test suite finished with **74 passed, 0 failed**, and the GitHub Actions Docker checks passed on both the push and pull request.

All nine Milestone 10 source/configuration files were merged via PR #12. Local `main` and `origin/main` were confirmed synchronized (`0` behind, `0` ahead), with a clean working tree.

**Scope boundary:** Docker build and synthetic API smoke tests passed in GitHub Actions. The real private model **has not** yet been exercised inside a running Docker container or validated for real-world production deployment. The MLflow registry and private model artifacts remain on the local computer, not in GitHub.

## 2. Starting point and constraints

Milestone 10 built on the frozen model and research-only serving interface from prior milestones:

- The frozen weighted PyTorch MLP transforms **29 raw application features** into **55 model inputs** after preprocessing and has architecture **55 → 64 → 32 → 1**.
- The fitted preprocessing pipeline, model checkpoint, and **Platt calibrator** must remain unchanged.
- Model construction uses CPU inference, `eval()` mode, and `torch.inference_mode()` through the existing inference implementation.
- The historical frozen policy identifies `FROZEN_PROVISIONAL_CANDIDATE` and `production_approved: false`.
- The **Month-7 final holdout was previously consumed once**. No additional holdout evaluation, retuning, or recalibration occurred in Milestone 10. The older frozen policy may still record the holdout as `SEALED`, reflecting its historical creation time; the current operational status is **CONSUMED**.
- The research API returns fraud risk scores, **not automatic account-approval or rejection decisions**.
- The operating threshold is approximately **0.04929577464788732**, and the maximum investigation fraction is **5% of a full evaluation cohort**. This is not a per-request guarantee for the FastAPI `/predict` endpoint.
- Private datasets, model weights, calibration files, and local tracking databases must not be committed to GitHub or embedded in the Docker image.

## 3. Delivered components

### 3.1 Docker build configuration

Created:

- `Dockerfile` — Python 3.11 slim-based API image; installs the project requirements, includes service code/configuration, uses a non-root user, and includes a `/ready` health check.
- `.dockerignore` — excludes Git metadata, local virtual environments, notebooks/tests, caches, data, artifacts, serialized model files, local MLflow databases, and potential secrets from Docker build context.

**Design choice:** The Docker image deliberately **does not contain the private trained model**. Running the real model in a container requires separately provisioning trusted, verified artifacts—for example, via a secure runtime mount—and then validating startup and inference. A successful build alone cannot establish that the real model runs inside Docker.

### 3.2 GitHub Actions CI

Created `.github/workflows/docker-ci.yml`, which validates Docker builds and performs dependency and synthetic API smoke checks. The workflow was updated for newer GitHub Actions versions and an explicit Ubuntu runner target.

Verified:

- **Push CI:** Successful for commit `b1a0b18` (workflow run #3; displayed duration approximately **2m 59s**).
- **Pull-request CI:** Successful on PR #12.
- **Merge compatibility:** No conflicts; PR #12 merged successfully.

**What CI does not prove:** The existing Docker workflow does not automatically perform the full **74-test pytest suite** or exercise real trained-model inference inside a container. Those are separate validation activities.

### 3.3 Frozen MLflow artifact bundle

Created `src/tracking/prepare_frozen_bundle.py` with **preview mode by default** and optional explicit `--commit` for creating a local bundle. It verifies file presence and the frozen binary artifact hashes before copying, then verifies copied files.

The bundle contains **eight files**:

| Artifact | Role |
|---|---|
| `configs/preprocessing_config.json` | Authoritative 29-feature configuration |
| `configs/milestone_07_candidate_policy_v01.json` | Frozen research-only policy |
| `artifacts/milestone_07/m07_weighted_mlp_m0to4_v01.json` | PyTorch checkpoint manifest |
| `artifacts/milestone_07/restart_bundle_v01/manifest.json` | Preprocessor/recovery manifest |
| `artifacts/milestone_07/calibration_policy_v01/manifest.json` | Calibration manifest |
| `artifacts/milestone_07/restart_bundle_v01/models.joblib` | Fitted preprocessing bundle |
| `artifacts/milestone_07/calibration_policy_v01/mlp_calibrators.joblib` | Platt calibrator |
| `artifacts/milestone_07/m07_weighted_mlp_m0to4_v01.pt` | Frozen PyTorch weights |

Core binary artifact size was approximately **0.19 MiB** total (rounded). The complete MLflow package is larger due to code, configuration, manifests, and environment files.

Private bundle path: `artifacts/milestone_10/frozen_bundle_v01/` (Git-ignored).

### 3.4 MLflow inference wrapper and schema

Created:

- `src/tracking/fraud_mlflow_model.py` — `FrozenFraudPyFunc`, a custom `mlflow.pyfunc.PythonModel` wrapper that loads the package's frozen bundle using `FraudInference` and returns calibrated `fraud_probability` scores.
- `src/tracking/model_signature.py` — explicit model signature generated from the feature configuration, with **18 numeric `double`**, **5 categorical `string`**, **6 binary `long`** inputs, and one `double` probability output.
- `src/tracking/export_frozen_mlflow_model.py` — preview-first local `mlflow.pyfunc.save_model()` export script with `artifacts`, `code_paths`, `signature`, and pinned package dependencies.

**Important distinction:** The MLflow signature declares column names/types, not all business rules. The FastAPI application independently validates binary 0/1 values and other request constraints. The MLflow PyFunc wrapper returns risk probabilities only—it does not apply an investigation-capacity policy or issue account decisions.

Export location: `artifacts/milestone_10/mlflow_export_v01/` (Git-ignored).

The export preserved and reloaded its own model code and artifacts, even when Python ran in isolated mode (`-I`) from another temporary working directory. This proves **local package portability**, not compatibility with every clean machine or platform.

### 3.5 Guarded MLflow Model Registry

Created `src/tracking/register_frozen_model.py`:

1. **Default preview:** validates the export and its packaged artifact checksums, computes a fingerprint, and inspects the existing SQLite registry read-only.
2. **Explicit commit:** creates/uses the Milestone 10 experiment, logs the saved model as an MLflow run artifact, and registers the run-linked model version.
3. **Duplicate guard:** refuses to create a registration when the target registered model/version already exists.
4. **Provenance tags:** record research-only status, consumed holdout, model type, calibration, feature count, and export fingerprint.

The script initially assumed the artifact directory was named `artifacts/frozen_bundle`. MLflow actually exported `artifacts/frozen_bundle_v01`, so the path resolver was corrected to **read the real location from the `MLmodel` metadata**. The corrected preview passed without database changes.

**Registered model:** `bank_account_fraud_weighted_mlp_frozen`  
**Version:** `1`  
**Experiment:** `bank-account-fraud-m10-registry`  
**Run ID:** `bddef72b580146aca721754fbb6ff5f6`  
**Run artifact URI:** `runs:/bddef72b580146aca721754fbb6ff5f6/frozen_fraud_pyfunc`  
**Registered model URI:** `models:/bank_account_fraud_weighted_mlp_frozen/1`  
**Tracking backend:** local SQLite database `artifacts/milestone_06/mlflow_tracking.db`  
**Export fingerprint (SHA-256):** `7b66d42ed3d91551ccf2a0bef453c927c29238481a53b4984236a55961e19ac2`

The model version, run, and experiment linkage were independently verified. The model was successfully loaded by its **`models:/.../1` URI** and its synthetic predictions matched the original frozen reference.

**Caveat:** A registry tag such as `production_approved=false` documents governance status; by itself it is **not an access-control or deployment-enforcement mechanism**. Production gating requires separate technical and organizational controls.

### 3.6 Automated tests

Added `tests/test_m10_mlflow_integration.py` with eight targeted tests:

1. 29-feature MLflow input/output schema validation.
2. Duplicate-feature detection.
3. PyFunc probability DataFrame contract.
4. Invalid non-DataFrame input rejection.
5. Valid synthetic artifact integrity verification.
6. Tampered-checkpoint detection.
7. Duplicate registered-model protection.
8. Preview-mode no-write behavior.

**Results:** Eight dedicated integration tests passed; the complete local suite passed with **74/74 tests** (`74 passed in 41.46s`). This suite primarily validates code and synthetic fixtures, not real-model inference in Docker.

## 4. Evidence and GitHub handoff

| Evidence | Verified result |
|---|---|
| Local registered-model load | `Step 10.67: PASS` |
| Local package portability check | `Step 10.60: PASS` |
| Complete pytest suite | `74 passed in 41.46s` |
| Git private-artifact audit | PASS; only three approved `.gitkeep` placeholders tracked in protected directories |
| GitHub push CI | Success on commit `b1a0b18` |
| GitHub PR CI | Success; two successful checks on PR #12 |
| Pull request | [#12](https://github.com/thekovakgroup/bank-account-fraud-dl-mlops/pull/12), merged and closed |
| Main merge commit | `89eff35` |
| Final local branch | `main`, clean; synchronized with `origin/main` (`0 0`) |
| Local registered model | Version 1 still present and research-only at final checkpoint |

**Nine files delivered in PR #12:**

```text
.dockerignore
.github/workflows/docker-ci.yml
Dockerfile
src/tracking/export_frozen_mlflow_model.py
src/tracking/fraud_mlflow_model.py
src/tracking/model_signature.py
src/tracking/prepare_frozen_bundle.py
src/tracking/register_frozen_model.py
tests/test_m10_mlflow_integration.py
```

**Preserved Git stashes — do not pop, apply, or delete without a separate plan:**

- `stash@{0}` — Milestone 07 notebook post-merge edits.
- `stash@{1}` — Milestone 04 additional evaluation and report cells.

## 5. Security, responsible AI, and limitations

**Implemented safeguards:**

- Private model weights, joblib bundles, datasets, and tracking database are excluded from Git and Docker build contexts.
- Frozen binary artifacts are protected by recorded SHA-256 checks; copied/exported artifacts are checked against source.
- MLflow export and registration are preview-first, with explicit write operations.
- Trusted local artifacts are loaded via model serialization; **never load untrusted CloudPickle/joblib model packages**, because deserialization can execute code.
- Existing read-only risk-scoring behavior is preserved; no automatic lending/account eligibility decision is made.
- `production_approved=False` and research-only tags are retained.
- The already consumed Month-7 holdout was **not** reused during this milestone.

**Known limitations and next work:**

- **Real-model Docker runtime not validated.** The CI container smoke test is synthetic. A separate secure runtime artifact mount and real-model container smoke test are needed.
- **Registry is local.** The MLflow SQLite database and private artifacts are not hosted, backed up as part of GitHub, or deployable solely from the public repository.
- **Cross-machine portability not yet validated.** A clean Python environment and dependency installation test remain necessary. The exported package records pinned dependencies, including a Torch version without the local `+cpu` suffix; CPU packaging/source selection should be validated separately.
- **No production authorization.** Tags are descriptive metadata; operational enforcement, security review, monitoring, privacy/compliance sign-off, and appropriate human oversight remain outstanding.
- **No new final-holdout testing.** Recorded model metrics should be cited from the already completed Milestone 07 report, not recalculated on Month 7.
- **MLflow integration tests are scoped.** The eight new tests do not cover all intermittent registry failures, retries, distributed storage behavior, or full Docker inference.

## 6. Handoff to Milestone 11

**Milestone 10 is complete.** Milestone 11 should focus on instructor-/interviewer-ready documentation and presentation, while preserving verified artifacts and the frozen evaluation protocol.

Suggested next work, in order:

1. Audit and update `README.md`, retaining existing Milestone 06–09 documentation and adding a precise Milestone 10 section.
2. Create an architecture diagram and explain the development, CI, registry, and API boundaries.
3. Summarize final recorded evaluation metrics by referencing `docs/milestone_07_final_holdout_report.md`; do not access the consumed Month-7 data again.
4. Prepare reproducibility instructions that explicitly distinguish public source code from private required artifacts.
5. Prepare an instructor-facing presentation, short demo narrative, and technical interview practice.
6. Finish with a clean Git privacy audit and final submission checklist.

---

**One-sentence interview takeaway:** *I took a frozen, calibrated PyTorch fraud-risk model and added secure Docker build validation, GitHub Actions CI, and a provenance-tracked, locally portable MLflow registered model—without retraining or reusing the consumed final holdout.*

# Milestone 10 — Technical Interview Questions and Model Answers

**Capstone:** Bank Account Fraud Detection — Deep Learning & MLOps  
**Scope:** Docker, CI/CD, MLflow artifact packaging, local Model Registry, validation, privacy, and responsible deployment  
**Milestone result:** PR #12 merged into `main` (`89eff35`); 74 local pytest tests passed; Docker CI passed; MLflow registered model `bank_account_fraud_weighted_mlp_frozen`, version 1  
**Model governance:** Research-only, `production_approved=False`; consumed Month-7 holdout not reused

Use these as **interview practice answers**. Explain the principles in your own words and distinguish what was *actually tested* from what would need to be tested before deployment.

## A. Project architecture and model serving

### 1. What was the objective of Milestone 10?

**Model answer:** I made an already frozen Deep Learning fraud-risk model easier to package, verify, and track. I added a Docker build definition, automated Docker validation in GitHub Actions, and a complete MLflow model export and local registered version. I did not retrain the model or retune its operating policy.

### 2. Describe the model pipeline end to end.

**Model answer:** A request supplies 29 raw application features. The existing fitted preprocessing pipeline generates 55 transformed inputs for a frozen weighted PyTorch MLP with layers 55 → 64 → 32 → 1. A frozen Platt calibrator converts model output into a calibrated fraud probability. The inference interface returns a risk score, not a bank-account decision.

### 3. Why keep preprocessing and calibration with the model?

**Model answer:** The checkpoint alone cannot reliably interpret raw applications. Model predictions depend on the exact fitted encoders/scalers and the calibrator learned earlier. Packaging them together avoids training-serving skew and makes the prediction pipeline reproducible.

### 4. What is the difference between model scoring and an investigation policy?

**Model answer:** Scoring estimates risk for each input. An investigation policy chooses which applications to review under operational constraints. Our frozen policy references a threshold of about 0.0493 and a 5% investigation capacity for a **full cohort**; the API and MLflow wrapper return probabilities and do not claim to enforce a global cohort budget on arbitrary individual request batches.

### 5. Why doesn't the model automatically reject account applications?

**Model answer:** Fraud scores are predictions, not final eligibility determinations. False positives, fairness, regulatory issues, operational capacity, and model limitations require appropriate human oversight and institution-specific governance. The milestone explicitly keeps the model research-only and unapproved for production.

## B. Docker and CI/CD

### 6. What is the difference between a Dockerfile and a Docker image?

**Model answer:** A Dockerfile is the declarative recipe for a container image. The image is the built artifact containing the runtime, dependencies, application code, and configuration. Running the image creates a container. In this project we validated image builds, but did not prove real-model container inference.

### 7. Why use `python:3.11-slim-bookworm` and a non-root user?

**Model answer:** The slim base helps reduce the image footprint, while a pinned Python family and Debian distribution make the runtime more predictable than an unspecified latest image. A non-root process reduces the impact of some container compromises, although it is only one security layer.

### 8. Why have both `.gitignore` and `.dockerignore`?

**Model answer:** `.gitignore` protects against accidentally staging local datasets, trained artifacts, and databases. `.dockerignore` separately prevents those files from entering the **Docker build context**, even if they exist locally. Neither is a replacement for secret management or secure runtime storage.

### 9. Why exclude the real model artifacts from the Docker image?

**Model answer:** The trained files and other private artifacts are not intended for public distribution through the repository or a published image. The container is built from shareable code; trusted private weights/preprocessor/calibrator must be supplied securely at runtime and verified before use.

### 10. What did GitHub Actions actually validate?

**Model answer:** It built the Docker image, checked installed dependencies, and ran a synthetic application/API smoke test in the container. The push check for `b1a0b18` and the pull-request check both succeeded. The full 74 pytest tests ran locally; the Docker workflow did not establish real-model inference inside the container.

### 11. What is CI versus CD in this project?

**Model answer:** Continuous integration automatically validated source changes when pushed or opened as a pull request. Continuous delivery/deployment would additionally release validated artifacts to a target runtime under controlled approvals. This milestone implemented CI-style Docker validation, **not automated production deployment**.

### 12. Why is a successful Docker build insufficient to claim production readiness?

**Model answer:** A build confirms the image can be assembled. It does not validate private artifact mounting, actual model startup, inference behavior in the container, resource usage, authentication, monitoring, operational risk, or institutional approvals. Those tests and controls remain future work.

## C. MLflow packaging and registry

### 13. What is MLflow, and which part did you use?

**Model answer:** MLflow supports managing machine-learning workflows and artifacts. I used its PyFunc model format and a **local MLflow tracking backend plus Model Registry** to store a versioned, run-linked frozen inference package with provenance metadata.

### 14. What is an MLflow `PythonModel` / PyFunc wrapper?

**Model answer:** It presents arbitrary model logic through MLflow's standard `load_context()` and `predict()` interface. My `FrozenFraudPyFunc` delegates to the existing `FraudInference` code, so I do not duplicate preprocessing, model loading, checksum verification, or calibration logic.

### 15. What does `load_context()` do in your wrapper?

**Model answer:** MLflow provides the resolved location of the packaged `frozen_bundle` artifact. The wrapper passes that location to `FraudInference`, which loads the fitted preprocessor, frozen PyTorch checkpoint, and calibrator after integrity checks. It also checks the research-only policy status.

### 16. What are the inputs and outputs of the MLflow model?

**Model answer:** The declared signature specifies **18 numeric doubles, 5 categorical strings, and 6 binary integer/long columns**, totaling 29 raw features. Output is a one-column DataFrame named `fraud_probability`, containing calibrated risk probabilities.

### 17. Does a model signature enforce all business validation rules?

**Model answer:** No. It describes required fields and MLflow types, but not necessarily restrictions such as only 0/1 for binary inputs, allowed categorical values, maximum request size, or application-specific semantics. The FastAPI layer applies stricter request validation separately.

### 18. What is the difference between an MLflow experiment, run, registered model, and model version?

**Model answer:** An **experiment** groups related tracking runs. A **run** records artifacts and metadata for one tracked operation. A **registered model** is a named collection of versions, and a **model version** identifies one registered artifact with a link to its source run. In this milestone version 1 is linked to a specific Milestone 10 run.

### 19. How do you load the registered model rather than the local export?

**Model answer:** I configure the local tracking/registry URI and load `models:/bank_account_fraud_weighted_mlp_frozen/1` with `mlflow.pyfunc.load_model()`. I verified two synthetic predictions against the original frozen inference engine with a tight numeric tolerance. That established functional local registry loading.

### 20. Why use a preview-first export/registration workflow?

**Model answer:** Preview mode verifies file presence, checksums, schema, package structure, and existing registry state without writing. An explicit `--commit` is required to create a bundle, export, or model version. That reduces accidental mutations and helps detect configuration errors before writing permanent state.

### 21. How did you prevent duplicate registration?

**Model answer:** The registration script inspects the local SQLite registry in read-only mode and refuses to register if the target model already exists. The corresponding test uses a temporary SQLite database. This is useful for a one-time controlled version-1 registration, but it is **not** a full concurrency-safe registration service or automated version-upgrade strategy.

### 22. What went wrong with the original registration preview, and how was it fixed?

**Model answer:** The script assumed MLflow would package an artifact under `artifacts/frozen_bundle`, but the export used `artifacts/frozen_bundle_v01`. I inspected the saved `MLmodel` metadata, then changed the script to resolve the recorded path rather than hard-code a directory name. The preview and subsequent registration succeeded.

### 23. What is the value of using a model URI such as `models:/.../1`?

**Model answer:** It identifies a registered version independently from a developer's original source directories. The version links to a run and its stored model package, improving provenance, repeatable loading, and review. It does **not** by itself mean the model is production-approved.

## D. Integrity, reproducibility, and security

### 24. Why compute SHA-256 checksums?

**Model answer:** Hash comparisons detect whether expected artifacts changed between training freeze, local packaging, and export. The export also has a package fingerprint to identify exactly what was registered. SHA-256 detects accidental or unauthorized changes relative to trusted expected hashes, but a hash alone is not a digital signature or access-control system.

### 25. What is the difference between a frozen model and a reproducible model package?

**Model answer:** A frozen model means its parameters and fitted components are not changing. A reproducible **inference** package additionally includes compatible code, preprocessing, calibration, schema, artifact identities, and dependency information so the same inputs produce consistent outputs. Reproducible inference is not the same as reproducing the original training from scratch.

### 26. What security concern does CloudPickle or joblib introduce?

**Model answer:** Python object deserialization can execute arbitrary code, so serialized models must be treated as executable and loaded only from trusted sources. My local MLflow export emitted a CloudPickle warning. The appropriate controls include trusted artifact storage, verification, controlled access, and avoiding untrusted model packages.

### 27. What does a clean GitHub repository prove—and what doesn't it prove?

**Model answer:** The Git audit showed that only intended source/configuration/test files were tracked; exported weights, joblib files, datasets, and the local MLflow SQLite database were not in the PR. This reduces inadvertent disclosure. It does not prove that a system is free from every secret or security vulnerability, and it does not back up the private local registry.

### 28. Was the model portable?

**Model answer:** It was portable **within the tested local environment**. I loaded the exported model in Python isolated mode from a different temporary directory and confirmed both its code and artifacts came from inside the MLflow export. I did not yet validate installation and inference on a fresh machine, a different Python version, or inside the real Docker runtime.

### 29. What does `production_approved=False` enforce?

**Model answer:** It is an explicit policy assertion in the frozen package and metadata, and the loading code checks its expected value. It **does not automatically stop** someone who can access the model artifacts from deploying them. Proper production gates require separate permissions, deployment approvals, monitoring, and governance.

## E. Evaluation and MLOps judgment

### 30. Why not use Month 7 again to test the MLflow model?

**Model answer:** Month 7 was the one-time final holdout and had already been consumed. Reusing it during packaging or repeated experimentation could undermine the independence of the final evaluation. Instead, I compared the registered model with the frozen reference on **synthetic inputs**, which tests integration consistency without producing new holdout-performance claims.

### 31. How do you distinguish integration testing from model evaluation?

**Model answer:** Integration tests check whether the components load, accept valid inputs, reject invalid inputs, and return consistent results. Model evaluation measures predictive quality using an appropriate dataset and metrics such as PR-AUC, ROC-AUC, calibration, and operational performance. My Milestone 10 synthetic tests address integration—not accuracy, fairness, or new generalization results.

### 32. Why were eight additional automated tests useful?

**Model answer:** They guard the 29-feature contract, wrapper behavior, duplicate-feature detection, artifact tampering checks, duplicate-registration prevention, and no-write preview behavior. They raised the complete suite to **74 passing tests**. The tests deliberately use synthetic data and temporary files to avoid modifying private artifacts or the live registry.

### 33. What would you improve next in the CI pipeline?

**Model answer:** I would add the complete automated pytest suite as a CI job and build a secure, controlled integration test that supplies private or non-sensitive test artifacts to a container at runtime. I'd also validate environment reproduction and artifact availability, rather than treating successful Docker builds as evidence of real-model serving.

### 34. What would be needed to move this from research-only to a production candidate?

**Model answer:** Beyond functional testing, I'd require documented data governance and privacy controls, security and dependency review, representative operational testing, model-risk assessment, bias/fairness analyses as appropriate, human-review procedures, monitoring and rollback plans, controlled artifact storage, secure deployment approvals, and institution-specific compliance sign-off. None of those approvals occurred in Milestone 10.

### 35. How would you summarize your specific technical contribution?

**Model answer:** I integrated the existing frozen model into MLflow's standard prediction interface, built an explicit signature from authoritative configuration, preserved the model's SHA-256-verified artifact bundle, implemented preview-first export/registration with duplicate protection, verified registered-model inference, and delivered Docker build validation through GitHub Actions. I kept the model and holdout protocol unchanged.

## F. Scenario-based interview questions

### 36. The Docker image builds, but `/ready` returns 503 with the private model absent. What do you do?

**Model answer:** I would treat this as expected when the private runtime artifacts are intentionally not mounted. I would verify the trusted artifact mount path, manifests, file access for the non-root process, and model startup logs; I would not copy private model files into public Git. I would only call the container ready after an authorized runtime integration test succeeds.

### 37. A packaged checkpoint hash differs from the frozen manifest. Should you regenerate the manifest?

**Model answer:** No. A mismatch means the packaged artifact cannot be trusted as the intended frozen checkpoint. I'd stop loading/registration, investigate source and copy paths, recover the known-good artifact, and verify hashes. Quietly updating the manifest would erase the integrity evidence.

### 38. `mlflow.pyfunc.load_model()` works locally but fails on a clean VM. What do you investigate?

**Model answer:** I'd check Python and library versions, especially PyTorch/CPU package selection; confirm packaged code and artifact paths; verify supported OS/system dependencies and model file permissions; and compare the clean VM environment to the exported requirements. Passing a local isolated-directory test does not guarantee cross-machine portability.

### 39. An interviewer says, “You passed CI, so the model is production ready.” How do you respond?

**Model answer:** CI proves that the checks we actually wrote passed. In this milestone, it established the Docker build and synthetic API behavior. Real-model container serving, security, monitoring, operational governance, and production sign-off remain unverified. I would not equate green CI with model approval.

### 40. A teammate proposes reusing the consumed Month-7 holdout to tune thresholds after registry integration. What do you say?

**Model answer:** I would reject further tuning on the consumed final holdout because that risks leakage from evaluation into development. If additional tuning is warranted, we need an appropriately separated development dataset and, when legitimate and available, a new untouched final evaluation protocol—not repeated optimization against Month 7.

## G. Short spoken responses to rehearse

### 30-second milestone explanation

> In Milestone 10, I operationalized a frozen PyTorch fraud-risk model without changing its training or calibration. I created secure Docker build rules and GitHub Actions CI, then packaged preprocessing, model weights, and calibration into a SHA-256-verified MLflow PyFunc model. I registered version 1 locally, verified its synthetic predictions against the frozen reference, and merged the changes after 74 local tests and successful CI checks. The model remains research-only, not production-approved.

### 90-second technical walkthrough

> The input to the service is 29 raw application features. A previously fitted preprocessing pipeline produces 55 features, which feed the frozen weighted PyTorch MLP. Its output goes through frozen Platt calibration to return a fraud probability. The inference code verifies hashes before loading the checkpoint, fitted preprocessing bundle, and calibrator. I built an MLflow `PythonModel` wrapper around that implementation and defined an explicit 29-column model signature. The model export includes code, dependencies, and an eight-file frozen artifact bundle. I tested that the export could load in an isolated local working directory and then registered it as version 1 in a local SQLite-backed MLflow registry. I retrieved the registered version by its `models:/` URI and checked its predictions against the frozen engine on synthetic applications. For software quality, I added Docker CI checks and eight MLflow-specific pytest tests, bringing the complete local suite to 74 passing tests. The actual trained model was intentionally excluded from GitHub and Docker build context. I would still need to validate real-model Docker serving, a clean-machine environment, and formal risk-governance controls before considering production use.

## H. Interview evidence you can cite

| Claim | Supporting project evidence |
|---|---|
| Registered model version 1 exists | Local MLflow Model Registry metadata and run `bddef72b580146aca721754fbb6ff5f6` |
| Registered inference works | Step 10.67 synthetic model-vs-reference comparison |
| Independent local package load works | Step 10.60 isolated-mode/local-directory check |
| Frozen binaries are integrity-checked | Frozen manifests and SHA-256 checks in `FraudInference` / Milestone 10 packaging scripts |
| Full local tests passed | `74 passed in 41.46s` |
| Docker build/smoke CI passed | GitHub Actions push and PR checks on PR #12 |
| Source is merged | PR #12; `main` merge commit `89eff35` |
| Production approval withheld | Frozen policy and MLflow version tags: `production_approved=false` |
| Month-7 holdout respected | No Milestone 10 holdout evaluation; status `CONSUMED` |

## I. Questions to ask your interviewer

1. How does your team separate model evaluation, deployment approval, and production rollout authority?
2. Which quality gates must a model pass before entering the organization's registry or serving environment?
3. How are private model artifacts stored, versioned, and securely injected into containers?
4. What observability do you expect for a risk-scoring API: data drift, calibration, latency, fairness, and error-rate monitoring?
5. How do data scientists, ML engineers, risk/compliance stakeholders, and platform teams jointly approve model changes?

---

**Practice rule:** Never claim you retrained the model, achieved a new Month-7 metric, validated real-model Docker inference, or obtained production approval during Milestone 10. State what the tests actually demonstrated and identify the remaining gates clearly.
