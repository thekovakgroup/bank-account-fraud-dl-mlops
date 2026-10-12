# Milestone 01 Architecture v1

## Project

Bank Account Fraud Detection - Deep Learning & MLOps

## System Goal

Build a reproducible fraud-risk scoring system for bank-account applications using the Bank Account Fraud Dataset Suite.

The system will support model development, evaluation, deployment, monitoring, and human review.

## High-Level Flow

```text
Bank Account Fraud Dataset Suite
        |
        v
Raw immutable Base.csv
        |
        v
Data validation and audit
        |
        v
Leakage review and split design
        |
        v
Training-only preprocessing
        |
        +----------------------+
        |                      |
        v                      v
Classical baseline        PyTorch MLP
        |                      |
        +----------+-----------+
                   |
                   v
         Validation comparison
                   |
                   v
      Calibration and threshold policy
                   |
                   v
          Sealed final test
                   |
                   v
          MLflow model registry
                   |
                   v
            FastAPI service
                   |
                   v
             Docker image
                   |
                   v
           GitHub Actions CI/CD
                   |
                   v
            Cloud deployment
                   |
                   v
        Drift and performance monitoring
                   |
                   v
             Human review workflow