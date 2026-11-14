"""Fungsi modeling & evaluasi yang dipakai bersama oleh semua halaman."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

LINEAR = "Linear Regression"
FOREST = "Random Forest Regressor"
MODEL_NAMES = [LINEAR, FOREST]


def build_model(name: str, **params):
    if name == LINEAR:
        return LinearRegression()
    if name == FOREST:
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            random_state=params.get("random_state", 42),
        )
    raise ValueError(f"Algoritma tidak dikenal: {name}")


def train_model(name: str, prep: dict, **params) -> Pipeline:
    """Latih pipeline (preprocessing + model) hanya dengan data training."""
    pipe = Pipeline(
        [
            ("preprocess", clone(prep["preprocessor"])),
            ("model", build_model(name, **params)),
        ]
    )
    pipe.fit(prep["X_train"], prep["y_train"])
    return pipe


def evaluate_model(model, X, y):
    """Kembalikan (dict metrik, array prediksi)."""
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    metrics = {
        "MAE": mean_absolute_error(y, pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y, pred),
    }
    return metrics, pred


def feature_effects(pipe: Pipeline, top_n: int = 15):
    """Importance (Random Forest) atau koefisien (Linear Regression)."""
    names = [n.split("__", 1)[1] for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        label, values = "Importance", est.feature_importances_
    else:
        label, values = "Koefisien", np.ravel(est.coef_)
    df = pd.DataFrame({"Fitur": names, label: values})
    df = df.reindex(df[label].abs().sort_values(ascending=False).index).head(top_n)
    return df.reset_index(drop=True), label
