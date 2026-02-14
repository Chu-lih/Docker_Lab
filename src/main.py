import argparse
import json
import os
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_dataset(name: str):
    name = name.lower().strip()
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name in {"breast_cancer", "cancer"}:
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from iris, wine, breast_cancer.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "breast_cancer"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--n-estimators", type=int, default=int(os.getenv("N_ESTIMATORS", "200")))
    parser.add_argument("--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "0")))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "artifacts/model.pkl"))
    parser.add_argument("--report-path", default=os.getenv("REPORT_PATH", "artifacts/report.json"))
    args = parser.parse_args()

    ds = load_dataset(args.dataset)
    X, y = ds.data, ds.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    max_depth = None if args.max_depth == 0 else args.max_depth

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    joblib.dump(model, args.model_path)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "test_size": args.test_size,
        "seed": args.seed,
        "model": "RandomForestClassifier",
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": max_depth,
        },
        "metrics": {
            "accuracy": acc,
            "f1_macro": f1,
        },
        "artifacts": {
            "model_path": args.model_path,
            "report_path": args.report_path,
        },
    }

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Training complete.")
    print(f"Saved model  -> {args.model_path}")
    print(f"Saved report -> {args.report_path}")
    print(f"Accuracy={acc:.4f}, F1_macro={f1:.4f}")
