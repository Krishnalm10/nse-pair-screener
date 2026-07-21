import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

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
# used as a fallback default universe if the rank-based CSV isn't available.
DEFAULT_VOLUME_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "SBIN", "TCS", "BAJFINANCE",
    "LT", "HINDUNILVR", "SUNPHARMA", "MARUTI", "ADANIPORTS", "INFY", "ADANIENT",
    "AXISBANK", "TITAN", "KOTAKBANK", "M&M", "ITC", "NTPC", "ULTRACEMCO",
    "HCLTECH", "BEL"
]

@st.cache_data
def load_rank_universe():
    """
    Loads NSE symbols ranked 300-1000 by market cap (a one-time snapshot,
    saved as market_cap_rank_300_1000.csv - refresh periodically by re-running
    the scraper against stockanalysis.com/list/nse-india/ if a current ranking matters).
    Falls back to the Nifty 50 list if the CSV isn't found.
    """
    try:
        df = pd.read_csv("market_cap_rank_300_1000.csv")
        return df["Symbol"].str.strip().tolist()
    except FileNotFoundError:
        return DEFAULT_VOLUME_UNIVERSE

# ============================================================
# 1-MINUTE VOLUME SCANNER FUNCTIONS
# ============================================================
# Design note: data fetching is isolated in _fetch_1min_bars_yfinance_batch().
# When a working 5paisa (or other broker) integration is ready, add a new
# _fetch_1min_bars_5paisa_batch() function and route to it via the data_source
# parameter below - the scanning/ratio logic itself won't need to change.

def _fetch_1min_bars_yfinance_batch(symbols, interval="1m"):
    """
    Fetches intraday bars for a list of NSE symbols via yfinance, at the given interval.
    NOTE: yfinance's 1-minute data only covers the last ~7 days (a hard limit of the
    free API). Longer intervals (2m/5m/15m) allow a longer lookback period, so we
    fetch more history for those to ensure enough bars for a meaningful rolling average.
    """
    # How much history to pull, chosen per interval to balance "enough bars for a
    # rolling average" against yfinance's own limits for each interval.
    period_by_interval = {
        "1m": "1d",
        "2m": "5d",
        "5m": "5d",
        "15m": "1mo",
    }
    period = period_by_interval.get(interval, "1d")

    tickers = [s + ".NS" for s in symbols]
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                        progress=False, threads=True)
    return tickers, data

# NSE cash market hours: 9:15 AM - 3:30 PM IST. The first and last 5 minutes
# naturally see structurally higher volume for every stock (opening/closing
# auctions, order accumulation) - including these would flag "unusual" volume
# that isn't actually unusual, just a normal feature of market open/close.
from datetime import time as _time
_MARKET_OPEN = _time(9, 15)
_EXCLUDE_OPEN_UNTIL = _time(9, 20)   # exclude 9:15:00 - 9:19:59
_EXCLUDE_CLOSE_FROM = _time(15, 25)  # exclude 15:25:00 - 15:30:00
_MARKET_CLOSE = _time(15, 30)

def _is_excluded_bar(timestamp):
    t = timestamp.time()
    if _MARKET_OPEN <= t < _EXCLUDE_OPEN_UNTIL:
        return True
    if _EXCLUDE_CLOSE_FROM <= t <= _MARKET_CLOSE:
        return True
    return False

def _filter_market_hours(df):
    """Drops bars in the first 5 and last 5 minutes of the trading session."""
    keep_mask = [not _is_excluded_bar(ts) for ts in df.index]
    return df[keep_mask]

def scan_unusual_volume_1min(symbols, lookback_bars=20, min_ratio=2.0, data_source="yfinance", interval="1m"):
    """
    Scans EVERY bar at the given interval (not just the latest one) for each symbol,
    comparing each bar's volume against the trailing average of the preceding
    lookback_bars bars. Returns every bar where the ratio meets or exceeds min_ratio -
    a stock can appear multiple times if it had several unusual spikes. Results are
    grouped by symbol (each stock's entries appear together, ordered by that stock's
    strongest spike first), with entries within a symbol sorted chronologically.
    Excludes the first/last 5 minutes of the trading session, since volume there is
    structurally elevated for every stock, not a genuine anomaly.
    """
    if data_source == "yfinance":
        tickers, data = _fetch_1min_bars_yfinance_batch(symbols, interval=interval)
    else:
        raise NotImplementedError(
            f"Data source '{data_source}' is not yet available. "
            f"Only 'yfinance' is currently supported - 5paisa integration is pending."
        )

    results = []
    for symbol, ticker in zip(symbols, tickers):
        try:
            stock_data = data if len(tickers) == 1 else data[ticker]
            stock_data = _filter_market_hours(stock_data)

            volume = stock_data["Volume"].dropna()
            close = stock_data["Close"].reindex(volume.index)

            if len(volume) < lookback_bars + 1:
                continue

            # Rolling average of the PRECEDING lookback_bars bars, for every bar in the day
            # (shift(1) excludes the current bar itself from its own average)
            rolling_avg = volume.shift(1).rolling(lookback_bars).mean()
            ratio_series = volume / rolling_avg

            prev_close = close.shift(1)
            pct_change_series = (close - prev_close) / prev_close * 100

            valid_mask = rolling_avg.notna() & (rolling_avg > 0)
            flagged_mask = valid_mask & (ratio_series >= min_ratio)

            for ts in volume.index[flagged_mask]:
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
                time_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else ""
                pct_val = pct_change_series.loc[ts]

                results.append({
                    "Symbol": symbol,
                    "_sort_ts": ts,  # internal, used only for chronological sort within a symbol
                    "Date": date_str,
                    "Bar Time": time_str,
                    "Volume": int(volume.loc[ts]),
                    f"{lookback_bars}-Bar Avg Volume": int(rolling_avg.loc[ts]),
                    "Volume Ratio": round(float(ratio_series.loc[ts]), 2),
                    "Bar Price Change %": round(float(pct_val), 3) if pd.notna(pct_val) else None,
                    "Close": round(float(close.loc[ts]), 2),
                })
        except Exception:
            continue  # skip symbols with missing/malformed data rather than failing the whole scan

    # Group entries by symbol (so a stock's multiple spikes appear together),
    # ordering symbols by their strongest (max) ratio first, and sorting each
    # symbol's own entries chronologically.
    max_ratio_by_symbol = {}
    for r in results:
        max_ratio_by_symbol[r["Symbol"]] = max(max_ratio_by_symbol.get(r["Symbol"], 0), r["Volume Ratio"])

    results.sort(key=lambda r: (-max_ratio_by_symbol[r["Symbol"]], r["Symbol"], r["_sort_ts"]))

    for r in results:
        del r["_sort_ts"]

    return results


st.title("NSE Research Tools")

symbol_df = load_symbol_list()
display_options = symbol_df["display"].tolist()

tab1, tab2, tab3 = st.tabs(["Pair Spread Support/Resistance", "Support & Resistance Finder", "Unusual Volume Scanner"])

# ============================================================
# TAB 1: PAIR SCREENER
# ============================================================

with tab1:
    st.write("Find historical support and resistance zones on the price spread between two NSE stocks.")

    col1, col2 = st.columns(2)
    with col1:
        symbol_a_display = st.selectbox("First stock", options=display_options,
                                         index=default_index(display_options, "SUNPHARMA"), key="pair_a")
    with col2:
        symbol_b_display = st.selectbox("Second stock", options=display_options,
                                         index=default_index(display_options, "DRREDDY"), key="pair_b")

    periods = ["1y", "2y", "3y", "5y", "10y", "20y"]

    chart_period = st.selectbox("Chart Period", options=periods,
                                 index=periods.index("2y"), key="pair_chart_period",
                                 help="Used for the spread chart and support/resistance zones below")

    pair_sr_order = 5       # fixed sensitivity - how many neighboring points confirm a swing point
    pair_sr_cluster = 0.5   # fixed cluster threshold (%) for grouping nearby raw swing points

    pair_sr_zone = st.slider("Zone width (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="pair_sr_zone")

    if st.button("Run Screening", key="run_screening"):
        symbol_a = extract_symbol(symbol_a_display) + ".NS"
        symbol_b = extract_symbol(symbol_b_display) + ".NS"

        try:
            df_chart = get_spread(symbol_a, symbol_b, chart_period)
            hedge_ratio = df_chart["hedge_ratio"].iloc[0]
            raw_hedge_ratio = df_chart["raw_hedge_ratio"].iloc[0]

            st.subheader(f"Spread Chart ({chart_period})")
            st.line_chart(df_chart["spread"])

            # --- Support/Resistance on the spread ---
            st.subheader("Support & Resistance on the Spread")
            st.caption("Zones as of today, using the full available history.")

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

            # --- Ratio-adjusted price comparison ---
            st.subheader("Ratio-Adjusted Price Comparison")
            df_chart["B_adjusted"] = df_chart["B_close"] * hedge_ratio
            st.line_chart(df_chart[["A_close", "B_adjusted"]])
            st.caption(f"B's price scaled by the rounded hedge ratio ({hedge_ratio}, raw was {raw_hedge_ratio:.4f}) so both lines are visually comparable. The gap between the lines is the spread shown above.")

        except Exception as e:
            st.error(f"Could not generate chart/S&R: {e}")

# ============================================================
# TAB 2: SUPPORT/RESISTANCE FINDER (single stock)
# ============================================================

with tab2:
    st.write("Find historical support and resistance zones for any NSE stock.")

    symbol_display = st.selectbox("Select stock", options=display_options,
                                   index=default_index(display_options, "RELIANCE"), key="sr_symbol")

    period_options = ["3mo", "6mo", "1y", "2y", "3y", "5y", "10y"]

    order = 5       # fixed sensitivity - how many neighboring points confirm a swing point
    cluster_pct = 0.5   # fixed cluster threshold (%) for grouping nearby raw swing points

    c1, c2 = st.columns(2)
    with c1:
        selected_period = st.selectbox("Time period", options=period_options, index=2, key="sr_period")
    with c2:
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
# TAB 3: UNUSUAL VOLUME SCANNER (1-Minute)
# ============================================================

with tab3:
    st.write("Scan every 1-minute bar of the trading day for NSE stocks trading at unusually high volume. A stock can appear multiple times if it had several unusual spikes during the day.")
    st.caption(
        "Currently powered by Yahoo Finance (yfinance), which only provides 1-minute data for "
        "the last ~7 days - this is a hard limit of the free data source, not adjustable. Data "
        "is also delayed (~15 min), not a true live feed. The first and last 5 minutes of the "
        "trading session (9:15-9:20 AM and 3:25-3:30 PM) are excluded, since volume there is "
        "structurally elevated for every stock, not a genuine anomaly. A broker API (e.g. 5paisa) "
        "integration is planned to replace this data source later without changing this tab's layout."
    )

    rank_universe = load_rank_universe()
    valid_rank_universe = [s for s in rank_universe if s in symbol_df["SYMBOL"].tolist()]

    st.caption(f"Stock pool: NSE rank 300-1000 by market cap ({len(valid_rank_universe)} stocks).")

    scan_all_ranked = st.checkbox(f"Scan all {len(valid_rank_universe)} stocks in this rank range (slower)", key="scan_all_ranked")

    if scan_all_ranked:
        selected_symbols_1min = valid_rank_universe
        st.write(f"Selected: all {len(valid_rank_universe)} stocks (rank 300-1000).")
    else:
        selected_symbols_1min = st.multiselect(
            "Or pick specific stocks to scan",
            options=valid_rank_universe,
            default=valid_rank_universe[:20],
            key="volume_symbols_1min"
        )

    vol1_col1, vol1_col2, vol1_col3 = st.columns(3)
    with vol1_col1:
        interval_1min = st.selectbox("Bar interval", options=["1m", "2m", "5m", "15m"], index=0, key="volume_interval_1min")
    with vol1_col2:
        lookback_bars = st.slider("Lookback period for average volume (bars)", min_value=5, max_value=60, value=20, key="volume_lookback_1min")
    with vol1_col3:
        min_ratio_1min = st.slider("Minimum volume ratio to flag as 'unusual'", min_value=1.2, max_value=5.0, value=2.0, step=0.1, key="volume_min_ratio_1min")

    if len(selected_symbols_1min) > 40:
        st.warning("Intraday data scans are heavier than daily scans. Large lists (40+ stocks) may be slow or hit rate limits - consider narrowing the list.")

    if st.button("Scan for Unusual Volume", key="run_volume_scan_1min"):
        if not selected_symbols_1min:
            st.warning("Select at least one stock to scan.")
        else:
            with st.spinner(f"Scanning {len(selected_symbols_1min)} stocks at {interval_1min} intervals..."):
                try:
                    results_1min = scan_unusual_volume_1min(
                        selected_symbols_1min,
                        lookback_bars=lookback_bars,
                        min_ratio=min_ratio_1min,
                        data_source="yfinance",
                        interval=interval_1min
                    )
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    results_1min = []

            if results_1min:
                st.success(f"Found {len(results_1min)} unusual-volume bar(s) across {len(selected_symbols_1min)} stock(s) (>= {min_ratio_1min}x the {lookback_bars}-bar average).")
                st.table(pd.DataFrame(results_1min))
            else:
                st.info("No stocks matched the unusual volume threshold with these settings.")