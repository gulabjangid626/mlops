import os
import pandas as pd
import yfinance as yf

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

os.makedirs("data/stocks", exist_ok=True)

all_frames = []

for ticker, name in TICKERS.items():
    try:
        raw = yf.Ticker(ticker)
        df  = raw.history(period="5y").reset_index()[["Date", "Close"]]
        df.columns = ["ds", "y"]
        df["ds"]     = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = ticker
        df["name"]   = name

        safe = ticker.replace(".", "_")
        df.to_csv(f"data/stocks/{safe}.csv", index=False)
        all_frames.append(df)
        print(f"✓ {ticker} — {name} — {len(df)} rows")
    except Exception as e:
        print(f"✗ {ticker} failed: {e}")

# Combined CSV — useful for exploratory analysis
combined = pd.concat(all_frames, ignore_index=True)
combined.to_csv("data/stocks/all.csv", index=False)

# Per-ticker reference / current split for drift detection
for ticker in TICKERS:
    safe   = ticker.replace(".", "_")
    single = pd.read_csv(
        f"data/stocks/{safe}.csv",
        parse_dates=["ds"]
    )[["ds", "y"]]

    cutoff = single["ds"].max() - pd.Timedelta(days=30)
    single[single["ds"] <= cutoff].tail(60).to_csv(
        f"data/stocks/{safe}_reference.csv", index=False
    )
    single[single["ds"] > cutoff].to_csv(
        f"data/stocks/{safe}_current.csv", index=False
    )

print(f"\n✅ {len(combined)} rows ingested across {len(TICKERS)} tickers.")