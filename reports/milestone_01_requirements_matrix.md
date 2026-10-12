# Milestone 01 Requirements Matrix

## Project

Bank Account Fraud Detection - Deep Learning & MLOps

## Purpose

Map the bootcamp syllabus requirements to concrete project deliverables so capstone coverage can be verified throughout development.

| Requirement | Planned Evidence | Milestone |
|---|---|---|
| Version-controlled Python project | GitHub repository, milestone branches, commit history | 01-12 |
| Reproducible project scaffold | Repository structure, `.gitignore`, environment documentation | 01 |
| Data ingestion and provenance | Data-source documentation and ingestion workflow | 01-02 |
| Exploratory data analysis | Data audit notebook, visualizations, summary reports | 02 |
| Leakage prevention | Leakage review and split-design report | 02 |
| Proper train/validation/test strategy | Documented split policy with sealed final test set | 02-07 |
| Classical baseline | Dummy classifier and logistic regression | 04 |
| Deep-learning model | PyTorch multilayer perceptron | 05-06 |
| Training/validation curves | PyTorch training history and figures | 05 |
| Imbalanced-data evaluation | PR-AUC, precision, recall, F1, ROC AUC, confusion matrix | 04-07 |
| Hyperparameter tuning | Documented tuning experiments | 06 |
| Experiment tracking | MLflow experiment tracking | 06 |
| Model registry | MLflow model registry and artifact metadata | 10 |
| Threshold policy | Validation-selected threshold with business-cost assumptions | 07 |
| Calibration | Calibration analysis and report | 07 |
| Final test evaluation | Frozen-model evaluation on sealed test set | 07 |
| Explainability | SHAP and/or LIME analysis | 08 |
| Fairness evaluation | Cohort metrics and false-positive disparity analysis | 08 |
| Security and privacy review | PII, input validation, logging, access-risk review | 08-11 |
| Responsible-AI guidance | Human-review policy, limitations, escalation guidance | 08 |
| Model serving | Documented FastAPI REST API | 09 |
| Input validation | Pydantic request schemas | 09 |
| Automated testing | Pytest suite for data, model, and API behavior | 01-10 |
| Containerization | Dockerfile and containerized API | 10 |
| CI/CD | GitHub Actions workflow | 10 |
| Cloud deployment | Deployed API on selected cloud platform | 11 |
| Drift monitoring | Evidently-based drift monitoring workflow | 11 |
| Cost awareness | Cloud and inference cost discussion | 11 |
| Architecture diagram | Data, model, serving, monitoring, and cloud architecture | 01 and 12 |
| Technical specification | Versioned capstone technical specification | 01 and 12 |
| Data card | Final data card | 12 |
| Model card | Final model card | 12 |
| README | Professional portfolio README | 01 and 12 |
| Screenshots and commands | Deployment/demo evidence | 09-12 |
| Error log and lessons learned | Documented implementation issues and resolutions | 01-12 |
| Cleanup instructions | Environment and cloud cleanup steps | 11-12 |
| Presentation | Final capstone presentation outline and demo script | 12 |
| Capstone defense | Interview questions and technical-defense answers | Every milestone and 12 |

## Capstone Success Standard

The project is complete only when the repository demonstrates the full lifecycle:

data
-> validation
-> preprocessing
-> baseline
-> deep learning
-> evaluation
-> responsible AI
-> model packaging
-> API serving
-> containerization
-> CI/CD
-> cloud deployment
-> monitoring
-> documentation
-> presentation