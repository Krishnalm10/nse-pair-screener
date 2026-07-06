import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller

@st.cache_data
def load_symbol_list():
    df = pd.read_csv("nse_equity_list.csv")
    df.columns = df.columns.str.strip()
    df["SYMBOL"] = df["SYMBOL"].str.strip().str.upper()
    df["NAME OF COMPANY"] = df["NAME OF COMPANY"].str.strip()
    df["display"] = df["SYMBOL"] + " - " + df["NAME OF COMPANY"]
    return df

def extract_symbol(display_string):
    return display_string.split(" - ")[0]

def get_spread(symbol_a, symbol_b, period):
    data_a = yf.download(symbol_a, period=period, interval="1d", progress=False)
    data_b = yf.download(symbol_b, period=period, interval="1d", progress=False)
    if data_a.empty or data_b.empty:
        raise ValueError("No data returned - check the symbol names")
    close_a = data_a["Close"].squeeze().rename("A_close")
    close_b = data_b["Close"].squeeze().rename("B_close")
    df = pd.concat([close_a, close_b], axis=1).dropna()
    df["spread"] = df["A_close"] / df["B_close"]
    return df

def run_adf_test(series):
    series = series.dropna()
    result = adfuller(series)
    return result[0], result[1]

def backtest_zscore_strategy(df, entry_threshold=2.0, exit_threshold=0.5):
    position = None
    entry_date = None
    entry_zscore = None
    entry_spread = None
    trades = []

    for date, row in df.iterrows():
        z = row["zscore"]
        spread = row["spread"]
        if pd.isna(z):
            continue

        if position is None:
            if z <= -entry_threshold:
                position, entry_date, entry_zscore, entry_spread = "long_spread", date, z, spread
            elif z >= entry_threshold:
                position, entry_date, entry_zscore, entry_spread = "short_spread", date, z, spread

        elif position == "long_spread" and z >= -exit_threshold:
            pct_return = (spread - entry_spread) / entry_spread * 100
            trades.append({"Type": "Long Spread", "Entry Date": entry_date.date(), "Exit Date": date.date(),
                            "Entry Z": round(entry_zscore, 2), "Exit Z": round(z, 2),
                            "Days Held": (date - entry_date).days, "P&L %": round(pct_return, 2)})
            position = None

        elif position == "short_spread" and z <= exit_threshold:
            pct_return = (entry_spread - spread) / entry_spread * 100
            trades.append({"Type": "Short Spread", "Entry Date": entry_date.date(), "Exit Date": date.date(),
                            "Entry Z": round(entry_zscore, 2), "Exit Z": round(z, 2),
                            "Days Held": (date - entry_date).days, "P&L %": round(pct_return, 2)})
            position = None

    return trades

st.set_page_config(page_title="NSE Pair Screener", layout="centered")
st.title("NSE Pair Stationarity Screener")
st.write("Test whether two NSE stocks have a statistically mean-reverting price spread.")

symbol_df = load_symbol_list()
display_options = symbol_df["display"].tolist()

def default_index(prefix):
    match = next((d for d in display_options if d.startswith(prefix + " ")), None)
    return display_options.index(match) if match else 0

col1, col2 = st.columns(2)
with col1:
    symbol_a_display = st.selectbox("First stock", options=display_options, index=default_index("SUNPHARMA"))
with col2:
    symbol_b_display = st.selectbox("Second stock", options=display_options, index=default_index("DRREDDY"))

periods = ["6mo", "1y", "2y", "3y", "5y"]

if st.button("Run Screening"):
    symbol_a_raw = extract_symbol(symbol_a_display)
    symbol_b_raw = extract_symbol(symbol_b_display)
    symbol_a = symbol_a_raw + ".NS"
    symbol_b = symbol_b_raw + ".NS"

    st.subheader(f"Stationarity Results: {symbol_a} vs {symbol_b}")

    results = []
    pass_count = 0
    for period in periods:
        try:
            df = get_spread(symbol_a, symbol_b, period)
            adf_stat, p_value = run_adf_test(df["spread"])
            is_stationary = p_value < 0.05
            if is_stationary:
                pass_count += 1
            results.append({
                "Period": period,
                "ADF Statistic": round(adf_stat, 4),
                "p-value": round(p_value, 4),
                "Stationary?": "✅ Yes" if is_stationary else "❌ No"
            })
        except Exception as e:
            results.append({"Period": period, "ADF Statistic": None, "p-value": None, "Stationary?": f"Failed: {e}"})

    st.table(pd.DataFrame(results))
    st.metric("Windows Passed", f"{pass_count} / {len(periods)}")

    if pass_count >= 3:
        st.success("Reasonably consistent evidence of mean-reversion. Worth further investigation.")
    elif pass_count >= 1:
        st.warning("Weak/inconsistent evidence. Likely not robust enough to trade on alone.")
    else:
        st.error("No evidence of mean-reversion across any window tested.")

    try:
        df_chart = get_spread(symbol_a, symbol_b, "2y")
        window = 20
        df_chart["spread_mean"] = df_chart["spread"].rolling(window).mean()
        df_chart["spread_std"] = df_chart["spread"].rolling(window).std()
        df_chart["zscore"] = (df_chart["spread"] - df_chart["spread_mean"]) / df_chart["spread_std"]

        st.subheader("Spread Chart (2 Year)")
        st.line_chart(df_chart[["spread", "spread_mean"]])

        st.subheader("Backtest: Z-score Mean-Reversion Strategy")
        st.caption("Hypothetical trades only - enters at |z| ≥ 2, exits when z reverts to within ±0.5. No real money, no transaction costs included.")

        trades = backtest_zscore_strategy(df_chart)

        if trades:
            trades_df = pd.DataFrame(trades)
            st.table(trades_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", len(trades_df))
            col2.metric("Avg Days Held", f"{trades_df['Days Held'].mean():.1f}")
            col3.metric("Avg P&L per Trade", f"{trades_df['P&L %'].mean():.2f}%")

            win_rate = (trades_df["P&L %"] > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}% ({(trades_df['P&L %'] > 0).sum()} of {len(trades_df)} trades profitable)")
        else:
            st.info("No trades would have triggered with these thresholds over this period.")

    except Exception as e:
        st.error(f"Could not generate chart/backtest: {e}")