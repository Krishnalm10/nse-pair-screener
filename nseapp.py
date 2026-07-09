import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import adfuller

# ============================================================
# SHARED FUNCTIONS (used by both tabs)
# ============================================================

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

def default_index(display_options, prefix):
    match = next((d for d in display_options if d.startswith(prefix + " ")), None)
    return display_options.index(match) if match else 0

# ============================================================
# PAIR SCREENER FUNCTIONS
# ============================================================

def round_hedge_ratio(ratio):
    """
    Round the hedge ratio's fractional part to a 'clean' trading-friendly value,
    keeping the whole-number part intact:
    - fractional part >0.8 or <0.2: round to nearest whole number (0 or 1, carries into whole part)
    - fractional part 0.35 to 0.65 (inclusive): round to nearest 0.5
    - otherwise (0.2-0.35 or 0.65-0.8): round to 0.25 if below 0.5, else 0.75
    """
    whole = int(ratio)  # ratio is always positive (price ratio), so this floors correctly
    frac = ratio - whole

    if frac > 0.8 or frac < 0.2:
        frac_rounded = round(frac)
    elif 0.35 <= frac <= 0.65:
        frac_rounded = round(frac * 2) / 2
    else:
        frac_rounded = 0.25 if frac < 0.5 else 0.75

    return whole + frac_rounded

def get_spread(symbol_a, symbol_b, period):
    data_a = yf.download(symbol_a, period=period, interval="1d", progress=False)
    data_b = yf.download(symbol_b, period=period, interval="1d", progress=False)
    if data_a.empty or data_b.empty:
        raise ValueError("No data returned - check the symbol names")
    close_a = data_a["Close"].squeeze().rename("A_close")
    close_b = data_b["Close"].squeeze().rename("B_close")
    df = pd.concat([close_a, close_b], axis=1).dropna()

    # Fixed hedge ratio calculated ONCE over the whole period (not per-day, which would cancel to zero)
    raw_hedge_ratio = df["A_close"].mean() / df["B_close"].mean()
    hedge_ratio = round_hedge_ratio(raw_hedge_ratio)
    df["hedge_ratio"] = hedge_ratio
    df["raw_hedge_ratio"] = raw_hedge_ratio
    df["spread"] = (df["A_close"] - (hedge_ratio * df["B_close"])).round(2)
    return df

def run_adf_test(series):
    series = series.dropna()
    result = adfuller(series)
    return result[0], result[1]

def backtest_walkforward(df, order, cluster_pct, zone_pct, entry_threshold=2.0,
                          exit_threshold=0.5, min_touches=2):
    """
    Walk-forward backtest: at each candidate entry date, zones are recalculated using
    ONLY data available up to and including that date - never future data. A zone only
    counts for confluence once it has accumulated MORE THAN min_touches touches based on
    what was knowable at that point in time. This avoids the look-ahead bias of computing
    zones once from the full period and applying them to earlier trades.
    """
    position = None
    entry_date = None
    entry_zscore = None
    entry_spread = None
    trades = []

    min_data_points = order * 2 + 5  # need enough history for argrelextrema to work meaningfully

    for i, (date, row) in enumerate(df.iterrows()):
        z = row["zscore"]
        spread = row["spread"]
        if pd.isna(z):
            continue

        if position is None:
            long_signal = z <= -entry_threshold
            short_signal = z >= entry_threshold

            if long_signal or short_signal:
                # Only use data up to and including today - nothing from the future
                historical_spread = df["spread"].iloc[:i + 1]

                if len(historical_spread) < min_data_points:
                    long_signal = short_signal = False
                else:
                    raw_support, raw_resistance = find_levels(
                        historical_spread, order=order, cluster_threshold_pct=cluster_pct
                    )
                    all_levels = raw_support + raw_resistance

                    support_candidates = [(p, c) for p, c in all_levels if p < spread]
                    resistance_candidates = [(p, c) for p, c in all_levels if p > spread]

                    merged_support = merge_overlapping_zones(support_candidates, zone_pct)
                    merged_resistance = merge_overlapping_zones(resistance_candidates, zone_pct)

                    # Only zones with MORE THAN min_touches count as valid for entry
                    strong_support = [(l, u, c) for l, u, c in merged_support if c > min_touches]
                    strong_resistance = [(l, u, c) for l, u, c in merged_resistance if c > min_touches]

                    long_signal = long_signal and in_any_zone(spread, strong_support)
                    short_signal = short_signal and in_any_zone(spread, strong_resistance)

            if long_signal:
                position, entry_date, entry_zscore, entry_spread = "long_spread", date, z, spread
            elif short_signal:
                position, entry_date, entry_zscore, entry_spread = "short_spread", date, z, spread

        elif position == "long_spread" and z >= -exit_threshold:
            pnl = round(spread - entry_spread, 2)
            trades.append({"Type": "Long Spread", "Entry Date": entry_date.date(), "Exit Date": date.date(),
                            "Entry Z": round(entry_zscore, 2), "Exit Z": round(z, 2),
                            "Days Held": (date - entry_date).days, "P&L (Rs.)": pnl})
            position = None
        elif position == "short_spread" and z <= exit_threshold:
            pnl = round(entry_spread - spread, 2)
            trades.append({"Type": "Short Spread", "Entry Date": entry_date.date(), "Exit Date": date.date(),
                            "Entry Z": round(entry_zscore, 2), "Exit Z": round(z, 2),
                            "Days Held": (date - entry_date).days, "P&L (Rs.)": pnl})
            position = None

    return trades

# ============================================================
# SUPPORT/RESISTANCE FUNCTIONS (shared by both tabs)
# ============================================================

def find_levels(series, order=5, cluster_threshold_pct=0.5):
    values = series.values
    max_idx = argrelextrema(values, np.greater, order=order)[0]
    min_idx = argrelextrema(values, np.less, order=order)[0]
    resistance_points = values[max_idx]
    support_points = values[min_idx]

    def cluster_levels(points):
        if len(points) == 0:
            return []
        points = sorted(points)
        clusters = [[points[0]]]
        for p in points[1:]:
            if abs(p - clusters[-1][-1]) / clusters[-1][-1] * 100 <= cluster_threshold_pct:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [(float(np.mean(c)), len(c)) for c in clusters]

    return cluster_levels(support_points), cluster_levels(resistance_points)

def format_zone(price, zone_pct):
    # Use abs(price) so this stays correct even for negative/near-zero values
    # (e.g. a rupee-denominated spread that oscillates around zero)
    half_width = abs(price) * (zone_pct / 100 / 2)
    return price - half_width, price + half_width

def merge_overlapping_zones(levels, zone_pct, max_zone_pct=None):
    if not levels:
        return []
    if max_zone_pct is None:
        max_zone_pct = zone_pct * 2

    levels = sorted(levels, key=lambda x: x[0])
    merged = []
    current_price, current_touches = levels[0]
    current_lower, current_upper = format_zone(current_price, zone_pct)
    zone_center = current_price

    for price, touches in levels[1:]:
        lower, upper = format_zone(price, zone_pct)
        potential_upper = max(current_upper, upper)
        width_if_merged = (potential_upper - current_lower) / zone_center * 100

        if lower <= current_upper and width_if_merged <= max_zone_pct:
            current_upper = potential_upper
            current_lower = min(current_lower, lower)
            current_touches += touches
        else:
            merged.append((current_lower, current_upper, current_touches))
            current_lower, current_upper = format_zone(price, zone_pct)
            current_touches = touches
            zone_center = price

    merged.append((current_lower, current_upper, current_touches))
    return merged

def cleanup_final_overlaps(zones):
    """
    Final pass: merge any zones in the final (already top-N filtered) list that still
    overlap in their displayed lower/upper bounds. Runs on a small, already-filtered
    list, so there's no risk of the runaway/transitive merging the cap was built to avoid.
    """
    if not zones:
        return []

    zones = sorted(zones, key=lambda z: z[0])
    cleaned = [zones[0]]

    for lower, upper, touches in zones[1:]:
        prev_lower, prev_upper, prev_touches = cleaned[-1]
        if lower <= prev_upper:  # still overlaps - merge
            cleaned[-1] = (min(prev_lower, lower), max(prev_upper, upper), prev_touches + touches)
        else:
            cleaned.append((lower, upper, touches))

    return cleaned

def in_any_zone(value, zones):
    """Check if a value falls within any (lower, upper, touches) zone."""
    return any(lower <= value <= upper for lower, upper, _ in zones)

# ============================================================
# UNUSUAL VOLUME SCANNER FUNCTIONS
# ============================================================

# A verified list of liquid, large-cap NSE stocks (Nifty 50 constituents as of Jul 2026)
# used as a sensible default universe - not the full NSE listing, to keep scans fast.
DEFAULT_VOLUME_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "SBIN", "TCS", "BAJFINANCE",
    "LT", "HINDUNILVR", "SUNPHARMA", "MARUTI", "ADANIPORTS", "INFY", "ADANIENT",
    "AXISBANK", "TITAN", "KOTAKBANK", "M&M", "ITC", "NTPC", "ULTRACEMCO",
    "HCLTECH", "BEL"
]

def scan_unusual_volume(symbols, lookback_days=20, min_ratio=2.0):
    """
    Batch-downloads volume data for a list of NSE symbols and flags any where
    the most recent day's volume exceeds min_ratio times the average of the
    preceding lookback_days. Uses a single batched yfinance call for speed.
    """
    tickers = [s + ".NS" for s in symbols]
    period = f"{lookback_days + 15}d"  # extra buffer days for weekends/holidays

    data = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                        progress=False, threads=True)

    results = []
    for symbol, ticker in zip(symbols, tickers):
        try:
            if len(tickers) == 1:
                stock_data = data
            else:
                stock_data = data[ticker]

            volume = stock_data["Volume"].dropna()
            close = stock_data["Close"].dropna()

            if len(volume) < lookback_days + 1:
                continue

            latest_volume = volume.iloc[-1]
            avg_volume = volume.iloc[-(lookback_days + 1):-1].mean()  # excludes latest day

            if avg_volume == 0 or pd.isna(avg_volume):
                continue

            ratio = latest_volume / avg_volume

            latest_close = close.iloc[-1]
            prev_close = close.iloc[-2] if len(close) >= 2 else latest_close
            pct_change = (latest_close - prev_close) / prev_close * 100 if prev_close else 0

            if ratio >= min_ratio:
                results.append({
                    "Symbol": symbol,
                    "Latest Volume": int(latest_volume),
                    f"{lookback_days}-Day Avg Volume": int(avg_volume),
                    "Volume Ratio": round(ratio, 2),
                    "Price Change %": round(pct_change, 2),
                    "Latest Close": round(float(latest_close), 2),
                })
        except Exception:
            continue  # skip symbols with missing/malformed data rather than failing the whole scan

    return sorted(results, key=lambda x: -x["Volume Ratio"])



st.set_page_config(page_title="NSE Research Tools", layout="centered")
st.title("NSE Research Tools")

symbol_df = load_symbol_list()
display_options = symbol_df["display"].tolist()

tab1, tab2, tab3 = st.tabs(["Pair Stationarity Screener", "Support & Resistance Finder", "Unusual Volume Scanner"])

# ============================================================
# TAB 1: PAIR SCREENER
# ============================================================

with tab1:
    st.write("Test whether two NSE stocks have a statistically mean-reverting price spread.")

    col1, col2 = st.columns(2)
    with col1:
        symbol_a_display = st.selectbox("First stock", options=display_options,
                                         index=default_index(display_options, "SUNPHARMA"), key="pair_a")
    with col2:
        symbol_b_display = st.selectbox("Second stock", options=display_options,
                                         index=default_index(display_options, "DRREDDY"), key="pair_b")

    periods = ["6mo", "1y", "2y", "3y", "5y"]

    chart_period = st.selectbox("Chart & Backtest Period", options=periods,
                                 index=periods.index("2y"), key="pair_chart_period",
                                 help="Used for the spread chart, support/resistance zones, and backtest below")

    if st.button("Run Screening", key="run_screening"):
        symbol_a = extract_symbol(symbol_a_display) + ".NS"
        symbol_b = extract_symbol(symbol_b_display) + ".NS"

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
                    "Stationary?": "Yes" if is_stationary else "No"
                })
            except Exception as e:
                results.append({"Period": period, "ADF Statistic": None, "p-value": None, "Stationary?": f"Failed: {e}"})

        st.table(pd.DataFrame(results))
        st.metric("Windows Passed", f"{pass_count} / {len(periods)}")

        if pass_count >= 3:
            st.success("Reasonably consistent evidence of mean-reversion.")
        elif pass_count >= 1:
            st.warning("Weak/inconsistent evidence.")
        else:
            st.error("No evidence of mean-reversion across any window tested.")

        try:
            df_chart = get_spread(symbol_a, symbol_b, chart_period)
            hedge_ratio = df_chart["hedge_ratio"].iloc[0]
            raw_hedge_ratio = df_chart["raw_hedge_ratio"].iloc[0]
            window = 20
            df_chart["spread_mean"] = df_chart["spread"].rolling(window).mean()
            df_chart["spread_std"] = df_chart["spread"].rolling(window).std()
            df_chart["zscore"] = (df_chart["spread"] - df_chart["spread_mean"]) / df_chart["spread_std"]

            st.subheader(f"Spread Chart ({chart_period})")
            st.line_chart(df_chart[["spread", "spread_mean"]])

            # --- Support/Resistance on the spread (computed BEFORE backtest, used for confluence) ---
            st.subheader("Support & Resistance on the Spread")
            st.caption("Zones as of today, using the full available history - shown for reference. The backtest below recalculates its own zones at each historical date using only data available at that time.")

            sr_col1, sr_col2, sr_col3 = st.columns(3)
            with sr_col1:
                pair_sr_order = st.slider("Sensitivity (order)", min_value=2, max_value=15, value=5, key="pair_sr_order")
            with sr_col2:
                pair_sr_cluster = st.slider("Cluster threshold (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="pair_sr_cluster")
            with sr_col3:
                pair_sr_zone = st.slider("Zone width (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="pair_sr_zone")

            current_spread = float(df_chart["spread"].iloc[-1])

            raw_support, raw_resistance = find_levels(df_chart["spread"], order=pair_sr_order, cluster_threshold_pct=pair_sr_cluster)
            all_spread_levels = raw_support + raw_resistance

            spread_support = [(p, c) for p, c in all_spread_levels if p < current_spread]
            spread_resistance = [(p, c) for p, c in all_spread_levels if p > current_spread]

            merged_spread_support = merge_overlapping_zones(spread_support, pair_sr_zone)
            merged_spread_support = sorted(merged_spread_support, key=lambda x: -x[2])[:5]
            merged_spread_support = cleanup_final_overlaps(merged_spread_support)
            merged_spread_support = sorted(merged_spread_support, key=lambda x: -x[2])

            merged_spread_resistance = merge_overlapping_zones(spread_resistance, pair_sr_zone)
            merged_spread_resistance = sorted(merged_spread_resistance, key=lambda x: -x[2])[:5]
            merged_spread_resistance = cleanup_final_overlaps(merged_spread_resistance)
            merged_spread_resistance = sorted(merged_spread_resistance, key=lambda x: -x[2])

            st.write(f"Current spread: Rs.{current_spread:.2f}  (hedge ratio: {hedge_ratio}, raw: {raw_hedge_ratio:.4f})")

            sr_table_col1, sr_table_col2 = st.columns(2)
            with sr_table_col1:
                st.write("**Support zones (spread):**")
                for lower, upper, touches in merged_spread_support:
                    st.write(f"Rs.{lower:.2f} - Rs.{upper:.2f} ({touches} touches)")
            with sr_table_col2:
                st.write("**Resistance zones (spread):**")
                for lower, upper, touches in merged_spread_resistance:
                    st.write(f"Rs.{lower:.2f} - Rs.{upper:.2f} ({touches} touches)")

            # --- Backtest, now requiring walk-forward confluence with the zones above ---
            st.subheader("Backtest: Z-score + Support/Resistance Confluence (Walk-Forward)")

            min_touches = st.slider("Minimum zone touches required before it counts", min_value=1, max_value=5, value=2, key="pair_min_touches")

            st.caption(
                "Hypothetical trades only. No real money, no transaction costs included. "
                "An entry only triggers when the z-score threshold AND the spread sitting inside "
                "a support/resistance zone agree (long only inside a support zone, short only inside "
                "a resistance zone)."
            )
            st.info(
                f"Zones are recalculated at each candidate entry date using ONLY data available up "
                f"to that date - never future data. A zone only counts once it has more than "
                f"{min_touches} touches based on what was knowable at that point in time. This avoids "
                f"the look-ahead bias of computing zones once from the full period."
            )

            trades = backtest_walkforward(
                df_chart,
                order=pair_sr_order,
                cluster_pct=pair_sr_cluster,
                zone_pct=pair_sr_zone,
                min_touches=min_touches
            )

            if trades:
                trades_df = pd.DataFrame(trades)
                st.table(trades_df)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trades", len(trades_df))
                c2.metric("Avg Days Held", f"{trades_df['Days Held'].mean():.1f}")
                c3.metric("Avg P&L per Trade", f"Rs.{trades_df['P&L (Rs.)'].mean():.2f}")

                win_rate = (trades_df["P&L (Rs.)"] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}% ({(trades_df['P&L (Rs.)'] > 0).sum()} of {len(trades_df)})")
            else:
                st.info("No trades triggered - z-score signal and support/resistance zones never aligned with these settings.")

            # --- Ratio-adjusted price comparison ---
            st.subheader("Ratio-Adjusted Price Comparison")
            df_chart["B_adjusted"] = df_chart["B_close"] * hedge_ratio
            st.line_chart(df_chart[["A_close", "B_adjusted"]])
            st.caption(f"B's price scaled by the rounded hedge ratio ({hedge_ratio}, raw was {raw_hedge_ratio:.4f}) so both lines are visually comparable. The gap between the lines is the spread shown above.")

        except Exception as e:
            st.error(f"Could not generate chart/backtest/S&R: {e}")

# ============================================================
# TAB 2: SUPPORT/RESISTANCE FINDER (single stock)
# ============================================================

with tab2:
    st.write("Find historical support and resistance zones for any NSE stock.")

    symbol_display = st.selectbox("Select stock", options=display_options,
                                   index=default_index(display_options, "RELIANCE"), key="sr_symbol")

    period_options = ["3mo", "6mo", "1y", "2y", "3y", "5y", "10y"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        order = st.slider("Sensitivity (order)", min_value=2, max_value=15, value=5, key="sr_order")
    with c2:
        cluster_pct = st.slider("Cluster threshold (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="sr_cluster")
    with c3:
        selected_period = st.selectbox("Time period", options=period_options, index=2, key="sr_period")
    with c4:
        zone_pct = st.slider("Zone width (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="sr_zone")

    if st.button("Find Levels", key="find_levels"):
        symbol = extract_symbol(symbol_display) + ".NS"
        st.subheader(f"Results: {symbol}")

        data = yf.download(symbol, period=selected_period, interval="1d", progress=False)
        if data.empty:
            st.warning(f"No data available for {selected_period}")
        else:
            close = data["Close"].squeeze()
            current_price = float(close.iloc[-1])

            raw_support, raw_resistance = find_levels(close, order=order, cluster_threshold_pct=cluster_pct)
            all_levels = raw_support + raw_resistance

            support_now = [(p, c) for p, c in all_levels if p < current_price]
            resistance_now = [(p, c) for p, c in all_levels if p > current_price]

            merged_support = merge_overlapping_zones(support_now, zone_pct)
            merged_support = sorted(merged_support, key=lambda x: -x[2])[:5]
            merged_support = cleanup_final_overlaps(merged_support)
            merged_support = sorted(merged_support, key=lambda x: -x[2])

            merged_resistance = merge_overlapping_zones(resistance_now, zone_pct)
            merged_resistance = sorted(merged_resistance, key=lambda x: -x[2])[:5]
            merged_resistance = cleanup_final_overlaps(merged_resistance)
            merged_resistance = sorted(merged_resistance, key=lambda x: -x[2])

            st.markdown(f"**{selected_period}** (current price: Rs.{current_price:.2f})")

            chart_col, table_col = st.columns([2, 1])
            with chart_col:
                st.line_chart(pd.DataFrame({"Close": close}))

            with table_col:
                st.write("**Support zones:**")
                for lower, upper, touches in merged_support:
                    st.write(f"Rs.{lower:.2f} - Rs.{upper:.2f} ({touches} touches)")

                st.write("**Resistance zones:**")
                for lower, upper, touches in merged_resistance:
                    st.write(f"Rs.{lower:.2f} - Rs.{upper:.2f} ({touches} touches)")

# ============================================================
# TAB 3: UNUSUAL VOLUME SCANNER
# ============================================================

with tab3:
    st.write("Scan for NSE stocks trading at unusually high volume compared to their recent average.")
    st.caption(
        "Data is delayed (~15 min via Yahoo Finance, not a live exchange feed). If run during "
        "market hours, today's volume is partial (market still open) and will look artificially "
        "low against full-day historical averages - results are most meaningful after market close."
    )

    all_symbols = symbol_df["SYMBOL"].tolist()

    selected_symbols = st.multiselect(
        "Stocks to scan (defaults to Nifty 50 constituents - add or remove as needed)",
        options=all_symbols,
        default=[s for s in DEFAULT_VOLUME_UNIVERSE if s in all_symbols],
        key="volume_symbols"
    )

    vol_col1, vol_col2 = st.columns(2)
    with vol_col1:
        lookback_days = st.slider("Lookback period for average volume (days)", min_value=5, max_value=60, value=20, key="volume_lookback")
    with vol_col2:
        min_ratio = st.slider("Minimum volume ratio to flag as 'unusual'", min_value=1.2, max_value=5.0, value=2.0, step=0.1, key="volume_min_ratio")

    if len(selected_symbols) > 75:
        st.warning("Large scans (75+ stocks) may be slow or hit rate limits. Consider narrowing the list.")

    if st.button("Scan for Unusual Volume", key="run_volume_scan"):
        if not selected_symbols:
            st.warning("Select at least one stock to scan.")
        else:
            with st.spinner(f"Scanning {len(selected_symbols)} stocks..."):
                try:
                    results = scan_unusual_volume(selected_symbols, lookback_days=lookback_days, min_ratio=min_ratio)
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    results = []

            if results:
                st.success(f"Found {len(results)} stock(s) with unusual volume (>= {min_ratio}x the {lookback_days}-day average).")
                st.table(pd.DataFrame(results))
            else:
                st.info("No stocks matched the unusual volume threshold with these settings.")