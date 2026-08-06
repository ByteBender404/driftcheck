# driftcheck

`driftcheck` is a Python CLI tool that detects statistical drift between two tabular datasets (CSV or Parquet).

## Problem Statement

When deploying machine learning models or analyzing data pipelines over time, the incoming data distribution can shift compared to the historical data. This phenomenon, known as **data drift**, can degrade model performance and cause silent failures. `driftcheck` helps you quickly identify which features have drifted significantly by generating a self-contained, easy-to-read HTML report.

## Installation

You can install `driftcheck` locally for development and usage:

```bash
git clone <repository_url>
cd driftcheck
pip install -e .
```

## Usage

Compare two datasets (CSV or Parquet) and generate an HTML report:

```bash
driftcheck compare old_sample.csv new_sample.csv --output report.html
```

You can optionally specify a subset of columns to check:

```bash
driftcheck compare old_sample.csv new_sample.csv --columns age,income,category --output report.html
```

## Sample Output

*(Screenshot placeholder - A clean HTML report with side-by-side distribution plots and a severity summary table)*
