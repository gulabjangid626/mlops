"""
push_model.py — Upload retrained Prophet models to HuggingFace Hub.

Security fix [F3]:
  Token is now read from the HF_TOKEN environment variable, which is
  injected by GitHub Actions from a repository secret.
  NEVER hardcode the token in source code.

  To set the secret:
    GitHub → repo Settings → Secrets and variables → Actions → New secret
    Name: HF_TOKEN   |   Value: your HuggingFace write token (hf_...)
"""

import os
import pickle
from datetime import datetime, timezone
from huggingface_hub import HfApi, ModelCard, ModelCardData
REPO_ID = "gulabjangid/Stock_forecast"

RUN_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

TICKERS = {
    "TCS.NS":        "Tata Consultancy Services",
    "RELIANCE.NS":   "Reliance Industries",
    "INFY.NS":       "Infosys",
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SBIN.NS":       "State Bank of India",
    "MARUTI.NS":     "Maruti Suzuki",
    "HCLTECH.NS":    "HCL Technologies",
}

# [F3] Read token from environment — raises a clear error if missing
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set.\n"
        "In GitHub Actions: add it as a repository secret.\n"
        "Locally: export HF_TOKEN=hf_..."
    )

api = HfApi(token=hf_token)

# Create repo once if it doesn't exist
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

pushed = []

for ticker, name in TICKERS.items():
    safe     = ticker.replace(".", "_")
    pkl_path = f"data/stocks/{safe}_model.pkl"

    if not os.path.exists(pkl_path):
        print(f"⚠  {ticker}: model pkl not found, skipping")
        continue

    hf_filename = f"{safe}.pkl"

    api.upload_file(
        path_or_fileobj=pkl_path,
        path_in_repo=hf_filename,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"chore: update {ticker} model — {RUN_DATE}",
        commit_description=(
            f"Automated retraining triggered by GitHub Actions.\n"
            f"- Ticker: {ticker} ({name})\n"
            f"- Run date: {RUN_DATE}\n"
            f"- Model: Prophet multiplicative seasonality\n"
            f"- Training data: 5 years of historical closing prices"
        ),
    )
    pushed.append(ticker)
    print(f"✓ Pushed {hf_filename} → {REPO_ID}")

# ── Update model card ─────────────────────────────────────────────────────────
ticker_list = "\n".join(f"- `{t}` — {n}" for t, n in TICKERS.items())
card_data   = ModelCardData(
    language="en",
    license="mit",
    library_name="prophet",
    tags=["time-series", "forecasting", "stock", "nse", "india"],
)
card_content = f"""---
{card_data.to_yaml()}
---

# Stock Forecast Models — NSE Multi-Ticker

Prophet time-series forecasting models trained on 5 years of historical
closing prices for 10 NSE blue-chip stocks.

## Supported Tickers

{ticker_list}

## Usage

```python
import pickle
from huggingface_hub import hf_hub_download
from prophet import Prophet

ticker = "TCS.NS"
safe   = ticker.replace(".", "_")
path   = hf_hub_download(repo_id="{REPO_ID}", filename=f"{{safe}}.pkl")

with open(path, "rb") as f:
    model = pickle.load(f)

future   = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))
```

## Pipeline

- Retrained daily via GitHub Actions at 02:00 AM IST.
- Metrics (RMSE, MAE, MAPE) logged to DagsHub MLflow.
- Last updated: {RUN_DATE}
"""

ModelCard(card_content).push_to_hub(REPO_ID, token=hf_token)
print(f"\n✅ Pushed {len(pushed)} model(s): {pushed}")
