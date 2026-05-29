import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.prophet
import dagshub
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

TICKERS = [
    "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "SBIN.NS", "MARUTI.NS", "HCLTECH.NS",
]

dagshub.init(
    repo_owner="gulabjangid",
    repo_name="Mlopbackend",
    mlflow=True
)
mlflow.set_experiment("stock-forecast-prophet")

# mlflow.set_experiment("stock-forecast-multi-ticker")

for ticker in TICKERS:
    safe = ticker.replace(".", "_")
    csv  = f"data/stocks/{safe}.csv"

    if not os.path.exists(csv):
        print(f"⚠  {ticker}: CSV not found, skipping")
        continue

    df = pd.read_csv(csv, parse_dates=["ds"])[["ds", "y"]].dropna()
    # Ensure timestamps have no time component — prevents alignment issues
    df["ds"] = df["ds"].dt.normalize()

    # ── Train / test split — hold out last 30 calendar days ─────────────────
    cutoff   = df["ds"].max() - pd.Timedelta(days=30)
    train_df = df[df["ds"] <= cutoff].copy()
    test_df  = df[df["ds"] >  cutoff].copy()
  
    with mlflow.start_run(run_name=f"prophet_{safe}"):
        mlflow.set_tag("ticker", ticker)
        mlflow.set_tag("model_type", "Prophet")
        mlflow.set_tag("seasonality_mode", "multiplicative")
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("changepoint_prior_scale", 0.05)
        mlflow.log_param("yearly_seasonality", True)
        mlflow.log_param("weekly_seasonality", True)

        # ── Fit on train split for evaluation ────────────────────────────────
        eval_model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        eval_model.fit(train_df)

        # ── Evaluate: generate enough future days and merge on nearest date ──
        if len(test_df) > 0:
            # Generate future covering the test window + buffer for weekends
            periods  = len(test_df) + 15
            future   = eval_model.make_future_dataframe(periods=periods, freq="B")
            forecast = eval_model.predict(future)

            # Normalize both sides to date-only, then merge — no KeyError
            forecast["ds"] = forecast["ds"].dt.normalize()
            test_df["ds"]  = test_df["ds"].dt.normalize()

            merged  = test_df.merge(
                forecast[["ds", "yhat"]],
                on="ds",
                how="inner"      # only rows that exist in both (trading days)
            )

            if len(merged) > 0:
                actuals = merged["y"].values
                preds   = merged["yhat"].values

                rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
                mae  = float(mean_absolute_error(actuals, preds))
                mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

                mlflow.log_metrics({
                    "rmse": round(rmse, 4),
                    "mae":  round(mae,  4),
                    "mape": round(mape, 4),
                    "eval_days": len(merged),
                })
                print(f"  {ticker} → RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  (n={len(merged)})")
            else:
                print(f"  ⚠ {ticker}: no overlapping dates after merge, skipping metrics")

        # ── Refit on FULL data — the model we actually save ──────────────────
        model_full = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model_full.fit(df)

        pkl_path = f"data/stocks/{safe}_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model_full, f)

        # Log model artifact + register in MLflow Model Registry
        mlflow.log_artifact(pkl_path, artifact_path="models")

        try:
            mlflow.prophet.log_model(
                pr_model=model_full,
                artifact_path="prophet_model",
                registered_model_name=f"stock-forecast-{safe}",
            )
            print(f"✓ {ticker} — model registered in MLflow registry")
        except Exception as e:
            # DagsHub free tier may not support full model registry — graceful fallback
            print(f"  ℹ {ticker} — registry skipped ({e}), artifact logged only")

        print(f"✓ {ticker} model saved → {pkl_path}")

print("\n✅ All tickers trained.")
