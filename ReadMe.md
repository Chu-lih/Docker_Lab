# Docker Lab 1 (My Version) - Model Training in a Container

## Overview
This lab demonstrates how to containerize an ML training script using Docker.
The container trains a scikit-learn RandomForest model and saves artifacts to an output folder.

## My Changes (Compared to the original lab)
- Added support for multiple datasets (default: `breast_cancer`, also supports `iris` and `wine`)
- Saved outputs to an `artifacts/` directory for clean volume mounting
- Added `report.json` containing metrics (accuracy, F1) and run configuration (dataset, params)

## Tech Stack
- Python 3.10
- scikit-learn + joblib
- Docker

## How to Build
From the Lab1 directory:

```bash
docker build -t mlops-docker-lab1:myv1 .
