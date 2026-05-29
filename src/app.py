from contextlib import asynccontextmanager
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import hf_hub_download
from prophet import Prophet
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID = os.environ.get("HF_REPO_ID")

# Local model directory — resolve in priority order:
#   1. MODEL_DIR env var (set this on any platform to a writable path)
#   2. Sibling `models/` directory next to this file (works locally)
#   3. System temp directory as last resort
_script_dir = Path(__file__).parent
_default_local = _script_dir / "models"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_default_local)))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_TICKERS: dict[str, str] = {
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

# In-memory model store — populated at startup
models: dict[str, Prophet] = {}


def _load_model(ticker: str) -> Prophet:
    """
    Load a Prophet model for the given ticker.

    Resolution order:
      1. Local file in MODEL_DIR (put pre-trained .pkl files here for dev)
      2. HuggingFace Hub download (production / CI) — version-aware cache,
         re-downloads only when a new commit is pushed to the repo.
    """
    safe      = ticker.replace(".", "_")
    local_pkl = MODEL_DIR / f"{safe}.pkl"

    # ── Priority 1: local file already present ────────────────────────────────
    if local_pkl.exists():
        print(f"  ✓ {ticker} — loaded from local file ({local_pkl})")
        with open(local_pkl, "rb") as f:
            return pickle.load(f)

    # ── Priority 2: download from HuggingFace Hub ─────────────────────────────
    # hf_hub_download manages its own version-aware cache at:
    #   $HF_HOME/hub/  (default ~/.cache/huggingface/hub/)
    # Override with HF_HOME env var on any platform that needs a custom path.
    # force_download=False (default) means: only re-download when the commit
    # hash on the Hub differs from the locally cached version.
    print(f"  ↓ {ticker} — downloading from {REPO_ID} ...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{safe}.pkl",
        force_download=False,
    )

    # Copy into MODEL_DIR so Priority 1 hits on the next startup
    # (only helps on platforms with a persistent filesystem — ephemeral
    #  platforms like Render free tier will re-download on next cold start,
    #  which is fine and expected)
    try:
        local_pkl.write_bytes(Path(cached_path).read_bytes())
        print(f"    → cached to {local_pkl}")
    except OSError:
        pass  # read-only filesystem (some serverless platforms) — not fatal

    with open(cached_path, "rb") as f:
        return pickle.load(f)


# ── Startup: eager-load all models ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"MODEL_DIR  : {MODEL_DIR}")
    print(f"HF_REPO_ID : {REPO_ID}")
    print(f"Loading {len(SUPPORTED_TICKERS)} models ...")

    for ticker in SUPPORTED_TICKERS:
        try:
            models[ticker] = _load_model(ticker)
        except Exception as e:
            # Don't crash the entire server if one model fails —
            # the /health endpoint will report which ones are missing
            print(f"  ✗ {ticker} failed: {e}")

    print(f"Ready — {len(models)}/{len(SUPPORTED_TICKERS)} models loaded.\n")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Forecast API",
    version="3.1.0",
    description=(
        "Prophet-based stock price forecasting for 10 NSE blue-chip stocks. "
        "Each ticker has its own dedicated model."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    ticker: str = Field(
        default="TCS.NS",
        description=f"NSE ticker symbol. Supported: {list(SUPPORTED_TICKERS.keys())}",
    )
    days: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_TICKERS:
            raise ValueError(
                f"Ticker '{v}' not supported. "
                f"Choose from: {list(SUPPORTED_TICKERS.keys())}"
            )
        return v


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "models_loaded": len(models),
        "model_dir": str(MODEL_DIR),
        "supported_tickers": list(SUPPORTED_TICKERS.keys()),
        "loaded_tickers": list(models.keys()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/tickers", tags=["Meta"])
def list_tickers():
    return [
        {"ticker": t, "name": n, "model_loaded": t in models}
        for t, n in SUPPORTED_TICKERS.items()
    ]


@app.post("/predict", response_model=list[ForecastPoint], tags=["Forecast"])
def predict(req: ForecastRequest):
    if req.ticker not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{req.ticker}' is not loaded. Check /health.",
        )
    model    = models[req.ticker]
    future   = model.make_future_dataframe(periods=req.days, freq="B")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(req.days).copy()
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        result[col] = result[col].round(2)
    return result.to_dict(orient="records")


@app.get("/historical", tags=["Data"])
def historical(ticker: str = "TCS.NS"):
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker '{ticker}' not supported.")

    safe     = ticker.replace(".", "_")
    csv_path = _script_dir / "data" / "stocks" / f"{safe}.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Historical data for '{ticker}' not found. Run ingest.py first.",
        )

    df = pd.read_csv(csv_path, parse_dates=["ds"])[["ds", "y"]].dropna().tail(90)
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    df["y"]  = df["y"].round(2)
    return {
        "ticker": ticker,
        "name": SUPPORTED_TICKERS[ticker],
        "data": df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
