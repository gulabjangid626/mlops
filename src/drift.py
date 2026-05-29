"""
drift.py — Two-layer drift detection for stock price forecasting pipelines.

Layer 1 — RETURN DISTRIBUTION DRIFT (Kolmogorov-Smirnov)
  Compares daily log-returns (not raw prices) between reference and current
  windows. Log-returns are stationary, so KS is a valid test here. A
  significant result means the volatility regime or return pattern has changed
  — which is a legitimate signal to retrain.

Layer 2 — MODEL RESIDUAL DRIFT (t-test on mean error)
  Loads the saved Prophet model for each ticker, runs it forward over the
  current window, and computes residuals (actual - predicted). Tests whether
  the mean residual has shifted significantly from zero using a one-sample
  t-test. If the model's errors are systematically biased, it needs retraining.

Pipeline gate logic:
  - RETRAIN signal → Layer 1 KS p < 0.05  AND  |mean residual| > threshold
  - Either layer alone is not enough — stocks can have short volatility spikes
    that don't invalidate the model, and small residual bias can be noise.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift

warnings.filterwarnings("ignore")

TICKERS = [
    "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "SBIN.NS", "MARUTI.NS", "HCLTECH.NS",
]

# Drift thresholds
KS_ALPHA          = 0.05    # p-value threshold for return distribution drift
RESIDUAL_ALPHA    = 0.05    # p-value threshold for residual mean t-test
RESIDUAL_BIAS_PCT = 2.0     # mean absolute residual must exceed 2% of price
                             # to qualify as model drift (filters noise)

os.makedirs("reports", exist_ok=True)

drift_summary = {}

for ticker in TICKERS:
    safe     = ticker.replace(".", "_")
    ref_path = Path(f"data/stocks/{safe}_reference.csv")
    cur_path = Path(f"data/stocks/{safe}_current.csv")
    mdl_path = Path(f"data/stocks/{safe}_model.pkl")

    if not ref_path.exists() or not cur_path.exists():
        print(f"⚠  {ticker}: reference/current CSV missing, skipping")
        continue

    ref_df = pd.read_csv(ref_path, parse_dates=["ds"])
    cur_df = pd.read_csv(cur_path, parse_dates=["ds"])

    # ── Normalise timestamps ──────────────────────────────────────────────────
    ref_df["ds"] = ref_df["ds"].dt.normalize()
    cur_df["ds"] = cur_df["ds"].dt.normalize()

    result = {
        "ticker": ticker,
        # Layer 1 defaults
        "return_drift_detected": False,
        "ks_p_value": None,
        "ref_return_mean": None,
        "cur_return_mean": None,
        "ref_return_std": None,
        "cur_return_std": None,
        # Layer 2 defaults
        "residual_drift_detected": False,
        "residual_t_pvalue": None,
        "mean_residual": None,
        "mean_residual_pct": None,
        # Combined gate
        "retrain_recommended": False,
    }

    # ════════════════════════════════════════════════════════════════════════
    # LAYER 1 — Log-return distribution drift (KS test)
    # ════════════════════════════════════════════════════════════════════════
    ref_returns = np.log(ref_df["y"] / ref_df["y"].shift(1)).dropna().values
    cur_returns = np.log(cur_df["y"] / cur_df["y"].shift(1)).dropna().values

    if len(ref_returns) >= 5 and len(cur_returns) >= 5:
        ks_stat, ks_pval = stats.ks_2samp(ref_returns, cur_returns)
        return_drift     = ks_pval < KS_ALPHA

        result.update({
            "return_drift_detected": bool(return_drift),
            "ks_p_value":            round(float(ks_pval), 6),
            "ref_return_mean":       round(float(ref_returns.mean()), 6),
            "cur_return_mean":       round(float(cur_returns.mean()), 6),
            "ref_return_std":        round(float(ref_returns.std()), 6),
            "cur_return_std":        round(float(cur_returns.std()), 6),
        })

        # ── Evidently report (visual) on returns, not raw prices ─────────────
        ref_ret_df = pd.DataFrame({"log_return": ref_returns})
        cur_ret_df = pd.DataFrame({"log_return": cur_returns})

        reference = Dataset.from_pandas(
            ref_ret_df,
            data_definition=DataDefinition(numerical_columns=["log_return"])
        )
        current = Dataset.from_pandas(
            cur_ret_df,
            data_definition=DataDefinition(numerical_columns=["log_return"])
        )
        report   = Report([DataDriftPreset(), ValueDrift(column="log_return")])
        snapshot = report.run(reference, current)
        snapshot.save_html(f"reports/drift_{safe}.html")
    else:
        print(f"  ⚠ {ticker}: not enough return data points for KS test")

    # ════════════════════════════════════════════════════════════════════════
    # LAYER 2 — Model residual drift (t-test on mean prediction error)
    # ════════════════════════════════════════════════════════════════════════
    if mdl_path.exists():
        try:
            with open(mdl_path, "rb") as f:
                model = pickle.load(f)

            # Generate forecast over the current window dates
            future   = model.make_future_dataframe(
                periods=len(cur_df) + 15, freq="B"
            )
            forecast = model.predict(future)
            forecast["ds"] = forecast["ds"].dt.normalize()

            merged = cur_df.merge(
                forecast[["ds", "yhat"]], on="ds", how="inner"
            )

            if len(merged) >= 3:
                residuals      = (merged["y"] - merged["yhat"]).values
                mean_residual  = float(residuals.mean())
                price_mean     = float(merged["y"].mean())
                residual_pct   = abs(mean_residual / price_mean) * 100

                # One-sample t-test: is mean residual significantly != 0?
                t_stat, t_pval = stats.ttest_1samp(residuals, popmean=0)

                # Both conditions must hold: statistically significant bias
                # AND economically meaningful (> RESIDUAL_BIAS_PCT of price)
                residual_drift = (t_pval < RESIDUAL_ALPHA) and (residual_pct > RESIDUAL_BIAS_PCT)

                result.update({
                    "residual_drift_detected": bool(residual_drift),
                    "residual_t_pvalue":       round(float(t_pval), 6),
                    "mean_residual":           round(mean_residual, 2),
                    "mean_residual_pct":       round(residual_pct, 2),
                })
            else:
                print(f"  ⚠ {ticker}: not enough overlapping dates for residual check")

        except Exception as e:
            print(f"  ⚠ {ticker}: residual check failed — {e}")
    else:
        print(f"  ℹ {ticker}: no model pkl found, skipping residual drift")

    # ── Combined gate: both layers must fire to recommend retrain ────────────
    result["retrain_recommended"] = (
        result["return_drift_detected"] and result["residual_drift_detected"]
    )

    drift_summary[ticker] = result

    # ── Console summary ───────────────────────────────────────────────────────
    l1 = "🔴" if result["return_drift_detected"]   else "🟢"
    l2 = "🔴" if result["residual_drift_detected"]  else "🟢"
    gate = "⚠ RETRAIN" if result["retrain_recommended"] else "✅ OK"

    ks_str  = f"KS p={result['ks_p_value']}"           if result["ks_p_value"]       is not None else "KS=N/A"
    res_str = f"residual={result['mean_residual_pct']}%" if result["mean_residual_pct"] is not None else "residual=N/A"

    print(f"  {gate}  {ticker}")
    print(f"    {l1} Layer 1 — Return drift:   {ks_str}")
    print(f"    {l2} Layer 2 — Residual drift: {res_str}")

# ── Write JSON summary for pipeline gate ─────────────────────────────────────
import json
summary_path = "reports/drift_summary.json"
with open(summary_path, "w") as f:
    json.dump(drift_summary, f, indent=2, default=str)

retrain_tickers = [t for t, v in drift_summary.items() if v.get("retrain_recommended")]
print(f"\n{'='*60}")
print(f"Tickers recommended for retrain: {retrain_tickers or 'None'}")
print(f"Drift summary written → {summary_path}")