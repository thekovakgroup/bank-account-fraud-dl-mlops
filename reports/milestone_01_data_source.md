# Milestone 01 Data Source Record

## Dataset

Bank Account Fraud Dataset Suite

## Initial File Selection

Primary file:

- `Base.csv`

The remaining suite files will not be used unless:

- the instructor requires comparison across the suite
- later analysis shows that additional files are necessary
- a documented modeling reason justifies their inclusion

## Target

Target column:

- `fraud_bool`

Expected task:

- binary classification

## Verified Scale

- exact row count: `1,000,000`
- exact column count: `32`
- target column: `fraud_bool`
- temporal field available for audit: `month`

## Required Verification After Download

Record:

- exact row count
- exact column count
- column names
- data types
- target values
- fraud prevalence
- file size
- missing-value counts
- duplicate counts
- unique-value counts for categorical fields
- distribution of `month`
- fraud rate by `month`

## Provenance Requirements

Document:

- source website
- dataset owner or publisher
- dataset version if available
- download date
- original filename
- license or usage terms
- any dataset documentation supplied by the publisher

## Raw Data Policy

The original `Base.csv` must remain immutable.

It is stored locally at:

```text
data/raw/Base.csv
```

The raw file must not be edited in place.

Any cleaned, transformed, sampled, or split datasets must be written to:

```text
data/processed/
```

## Git Policy

Raw and processed datasets must not be committed to Git.

The repository should contain only:

- provenance documentation
- processing code
- lightweight reports
- schema information
- checksums or manifests

## Integrity Check

Raw file:

- `data/raw/Base.csv`

SHA-256:

- `7BF10A37CE07E72E14C1B09E5EFEE3D27261BAFF4FACC7DA767B0474DCF9B809`

This checksum will be used to verify that the raw dataset remains unchanged.

## Current Source Information

Original archive:

- `archive.zip`

Archive contents:

- `Base.csv`
- `Variant I.csv`
- `Variant II.csv`
- `Variant III.csv`
- `Variant IV.csv`
- `Variant V.csv`

Selected file for this capstone:

- `Base.csv`

Raw file size:

- `213,427,735 bytes`

## Status

Data acquisition for `Base.csv` is complete.

Next required verification:

- exact dataset shape
- schema
- target prevalence
- missing values
- duplicate counts
- categorical cardinality
- `month` distribution
