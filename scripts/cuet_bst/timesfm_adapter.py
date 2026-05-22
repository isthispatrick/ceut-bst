from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np


def forecast_series(values: list[float], horizon: int = 5) -> dict[str, Any]:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if len(clean) < 2:
        return {
            "method": "insufficient_data",
            "forecast": [],
            "lower": [],
            "upper": [],
            "note": "Take at least two mocks before forecasting your trend.",
        }
    if os.getenv("CUET_USE_TIMESFM", "false").strip().lower() == "true":
        try:
            return _timesfm_forecast(clean, horizon)
        except Exception as exc:
            fallback = _fallback_forecast(clean, horizon)
            fallback["timesfm_error"] = str(exc)
            return fallback
    return _fallback_forecast(clean, horizon)


def _fallback_forecast(values: list[float], horizon: int) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    x = np.arange(len(arr), dtype=float)
    if len(arr) >= 3:
        slope, intercept = np.polyfit(x, arr, 1)
    else:
        slope = arr[-1] - arr[-2]
        intercept = arr[-1] - slope * (len(arr) - 1)
    alpha = 0.55
    ema = arr[0]
    for value in arr[1:]:
        ema = alpha * value + (1 - alpha) * ema
    future_x = np.arange(len(arr), len(arr) + horizon, dtype=float)
    trend = intercept + slope * future_x
    forecast = 0.55 * trend + 0.45 * ema
    residual = arr - (intercept + slope * x)
    spread = max(float(np.std(residual)), 4.0)
    return {
        "method": "local_ewma_linear_fallback",
        "forecast": np.clip(forecast, 0, 100).round(2).tolist(),
        "lower": np.clip(forecast - spread, 0, 100).round(2).tolist(),
        "upper": np.clip(forecast + spread, 0, 100).round(2).tolist(),
        "note": "TimesFM is disabled or unavailable, so this uses a local trend + moving-average forecast.",
    }


def _timesfm_forecast(values: list[float], horizon: int) -> dict[str, Any]:
    model, timesfm = _load_timesfm()
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[np.array(values, dtype=np.float32)],
    )
    point = np.clip(point_forecast[0], 0, 100)
    lower = point
    upper = point
    if quantile_forecast is not None and len(quantile_forecast.shape) == 3 and quantile_forecast.shape[-1] >= 10:
        lower = np.clip(quantile_forecast[0, :, 1], 0, 100)
        upper = np.clip(quantile_forecast[0, :, -1], 0, 100)
    return {
        "method": "timesfm_2p5",
        "forecast": point.round(2).tolist(),
        "lower": lower.round(2).tolist(),
        "upper": upper.round(2).tolist(),
        "note": "Forecast generated with optional TimesFM 2.5. Treat it as a study trend, not an exam prediction.",
        "model": os.getenv("CUET_TIMESFM_MODEL", "google/timesfm-2.5-200m-pytorch"),
    }


@lru_cache(maxsize=1)
def _load_timesfm():
    import timesfm

    try:
        import torch

        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    model_name = os.getenv("CUET_TIMESFM_MODEL", "google/timesfm-2.5-200m-pytorch")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(os.getenv("CUET_TIMESFM_MAX_CONTEXT", "1024")),
            max_horizon=int(os.getenv("CUET_TIMESFM_MAX_HORIZON", "32")),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model, timesfm
