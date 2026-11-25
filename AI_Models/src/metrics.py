"""
Standard METR-LA masked evaluation metrics (MAE, RMSE, MAPE, R2).
Missing sensor readings (ground truth == 0.0) are masked out per DCRNN benchmark protocol.
"""
import numpy as np


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = y_true > null_val
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = y_true > null_val
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))


def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, min_speed: float = 1.0) -> float:
    mask = y_true >= min_speed
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = y_true > null_val
    if not np.any(mask):
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_multi_horizon(y_true: np.ndarray, y_pred: np.ndarray, horizon_names=None) -> dict:
    """
    Evaluate predictions of shape (N, num_horizons).
    Returns per-horizon metrics and overall macro average.
    """
    if horizon_names is None:
        horizon_names = ["15 min", "30 min", "60 min"]

    results = {"horizons": {}, "overall": {}}
    maes, rmses, mapes, r2s = [], [], [], []

    for idx, name in enumerate(horizon_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        m = {
            "mae": round(masked_mae(yt, yp), 4),
            "rmse": round(masked_rmse(yt, yp), 4),
            "mape": round(masked_mape(yt, yp), 4),
            "r2": round(masked_r2(yt, yp), 4),
        }
        results["horizons"][name] = m
        maes.append(m["mae"])
        rmses.append(m["rmse"])
        mapes.append(m["mape"])
        r2s.append(m["r2"])

    results["overall"] = {
        "mae": round(float(np.mean(maes)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "mape": round(float(np.mean(mapes)), 4),
        "r2": round(float(np.mean(r2s)), 4),
    }
    return results
