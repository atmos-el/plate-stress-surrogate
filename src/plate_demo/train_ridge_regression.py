from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

""" NOTE：①  """
FEATURES = [
    "width",
    "height",
    "hole_radius",
    "hole_center_x",
    "hole_center_y",
    "thickness",
    "edge_load",
]
TARGETS = ["max_displacement", "max_von_mises"]
""" END """

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate models for plate dataset")
    parser.add_argument("--dataset", default="data/dataset.csv", help="dataset csv path")
    parser.add_argument("--metrics", default="outputs/scalar_prediction/metrics.json", help="metrics output path")
    parser.add_argument("--model-dir", default="outputs/scalar_prediction/models", help="temporary model output directory")
    parser.add_argument("--plot-dir", default="outputs/scalar_prediction/plots", help="plot output directory")
    return parser.parse_args()


def train_target(
    frame: pd.DataFrame,
    target: str,
    model_dir: Path,
    plot_dir: Path,
) -> tuple[dict[str, float], dict[str, list[float]]]:

    """ NOTE：②  """
    x = frame[FEATURES]
    y = frame[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
    """ END """

    """ NOTE：③  """
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    predictor = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    predictor.fit(x_train, y_train)
    """ END """

    pred = predictor.predict(x_test)

    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    """ NOTE：④  """
    joblib.dump(predictor, model_dir / f"{target}.joblib")

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    """ END """

    preview = {
        "actual": [float(v) for v in y_test.iloc[:5]],
        "predicted": [float(v) for v in pred[:5]],
    }
    render_plots(target, y_test, pred, plot_dir)
    return metrics, preview


def render_plots(target: str, y_test: pd.Series, pred: list[float], plot_dir: Path) -> None:
    actual = y_test.to_numpy()
    predicted = pred
    residual = predicted - actual

    scatter_path = plot_dir / f"{target}_parity.png"
    hist_path = plot_dir / f"{target}_residuals.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.8, edgecolors="none")
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{target}: parity plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=12, alpha=0.85)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title(f"{target}: residual distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    metrics_path = Path(args.metrics)
    model_dir = Path(args.model_dir)
    plot_dir = Path(args.plot_dir)

    frame = pd.read_csv(dataset_path)
    results: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        metrics, preview = train_target(frame, target, model_dir, plot_dir)
        results[target] = {"metrics": metrics, "preview": preview}
        print(target, metrics)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
