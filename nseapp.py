import os
import time
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

# ============================================================
# QUARTERLY EARNINGS REACTION FUNCTIONS
# ============================================================
# NOTE: this presents FACTUAL data only - EPS surprise and historical price/volume
# reaction. It deliberately does NOT produce a buy/sell recommendation; quarterly
# earnings reactions are notoriously hard to predict from a simple rule (a beat can
# still crash a stock on weak guidance, a miss can rally on relief) - the person
# reviewing this data should draw their own conclusion, not the app.

BULK_EARNINGS_FILE = "quarterly_earnings_reports_top500.csv"

@st.cache_data
def load_bulk_earnings_csv():
    """Loads the pre-fetched top-500 quarterly earnings CSV, if present. Returns None if not found."""
    if not os.path.exists(BULK_EARNINGS_FILE):
        return None
    try:
        df = pd.read_csv(BULK_EARNINGS_FILE)
        return df
    except Exception:
        return None

@st.cache_data
def load_nifty_long_history():
    """
    Fetches Nifty 50's full history ONCE (cached, shared across every stock) so
    we can compute each stock's excess reaction without re-fetching Nifty per stock.
    """
    try:
        hist = yf.Ticker("^NSEI").history(period="10y")
        dates = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
        hist = hist.set_axis(dates)
        return hist, dates
    except Exception:
        return None, None

def _nifty_reaction_for_date(nifty_hist, nifty_dates, report_date_naive):
    """Nifty 50's % move over the same before/after trading-day window as a given report date."""
    before_dates = nifty_dates[nifty_dates < report_date_naive]
    after_dates = nifty_dates[nifty_dates >= report_date_naive]
    if len(before_dates) == 0 or len(after_dates) == 0:
        return None
    pre_date = before_dates[-1]
    post_date = after_dates[0]
    try:
        pre_close = float(nifty_hist.loc[pre_date, "Close"])
        post_close = float(nifty_hist.loc[post_date, "Close"])
        return (post_close - pre_close) / pre_close * 100 if pre_close else None
    except Exception:
        return None

def get_earnings_reaction_from_bulk(symbol, bulk_df, nifty_hist, nifty_dates, num_quarters):
    """
    Reconstructs earnings reaction data for a symbol from the pre-fetched bulk CSV,
    computing Nifty 50 reaction / excess reaction on the fly against the cached
    long-range Nifty history. Returns None if the symbol isn't in the bulk file.
    """
    if bulk_df is None or nifty_hist is None:
        return None

    rows = bulk_df[bulk_df["Symbol"] == symbol].copy()
    if rows.empty:
        return None

    rows["_report_date_obj"] = pd.to_datetime(rows["Report Date"])
    rows = rows.sort_values("_report_date_obj", ascending=False).head(num_quarters)

    results = []
    for _, row in rows.iterrows():
        report_date_naive = row["_report_date_obj"]
        nifty_reaction = _nifty_reaction_for_date(nifty_hist, nifty_dates, report_date_naive)
        price_reaction = row["Price Reaction %"] if pd.notna(row["Price Reaction %"]) else None
        excess = (
            round(price_reaction - nifty_reaction, 2)
            if price_reaction is not None and nifty_reaction is not None
            else None
        )
        results.append({
            "Report Date": row["Report Date"],
            "_report_date_obj": report_date_naive,
            "EPS Estimate": row["EPS Estimate"] if pd.notna(row["EPS Estimate"]) else None,
            "Reported EPS": row["Reported EPS"] if pd.notna(row["Reported EPS"]) else None,
            "EPS Surprise %": row["EPS Surprise %"] if pd.notna(row["EPS Surprise %"]) else None,
            "Price Reaction %": price_reaction,
            "Nifty 50 Reaction %": round(nifty_reaction, 2) if nifty_reaction is not None else None,
            "Excess Reaction %": excess,
            "Volume vs 20d Avg": row["Volume vs 20d Avg"] if pd.notna(row["Volume vs 20d Avg"]) else None,
        })

    return results if results else None


def get_earnings_reaction(symbol, num_quarters=8):
    """
    Returns a list of past earnings dates for the symbol with EPS estimate/actual/
    surprise, the stock's actual price and volume reaction, the Nifty 50's reaction
    over the same window (to isolate stock-specific moves from market-wide moves),
    and how each quarter's reaction compares statistically to the stock's own history.
    Returns (results, error_message) - error_message is None on success.
    """
    ticker_symbol = symbol + ".NS"
    try:
        ticker = yf.Ticker(ticker_symbol)
        earnings = None
        last_error = None
        for attempt in range(3):
            try:
                earnings = ticker.get_earnings_dates(limit=num_quarters + 6)  # extra buffer, some rows are future/upcoming
                break
            except Exception as retry_error:
                last_error = retry_error
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))  # brief backoff before retrying
        if earnings is None:
            raise last_error
    except Exception as e:
        return None, f"Could not fetch earnings dates after 3 attempts: {e}"

    if earnings is None or earnings.empty:
        return None, "No earnings date data available for this symbol on yfinance."

    # Keep only PAST reports (future/upcoming estimates have no Reported EPS yet)
    if "Reported EPS" not in earnings.columns:
        return None, "Earnings data for this symbol doesn't include reported EPS figures."

    past_earnings = earnings.dropna(subset=["Reported EPS"]).sort_index(ascending=False).head(num_quarters)

    if past_earnings.empty:
        return None, "No past reported earnings found for this symbol."

    # Fetch daily price/volume history spanning all the report dates, with padding
    start = (past_earnings.index.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (past_earnings.index.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    try:
        hist = ticker.history(start=start, end=end)
    except Exception as e:
        return None, f"Could not fetch price history: {e}"

    if hist.empty:
        return None, "No price history available to measure reaction."

    hist_dates = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    hist = hist.set_axis(hist_dates)

    # Fetch Nifty 50 over the same window, to isolate stock-specific reaction
    # from broader market moves on the same day.
    try:
        nifty_hist = yf.Ticker("^NSEI").history(start=start, end=end)
        nifty_dates = nifty_hist.index.tz_localize(None) if nifty_hist.index.tz is not None else nifty_hist.index
        nifty_hist = nifty_hist.set_axis(nifty_dates)
    except Exception:
        nifty_hist = None
        nifty_dates = None

    results = []
    for report_date, row in past_earnings.iterrows():
        try:
            report_date_naive = report_date.tz_localize(None) if report_date.tzinfo is not None else report_date

            before_dates = hist_dates[hist_dates < report_date_naive]
            after_dates = hist_dates[hist_dates >= report_date_naive]

            if len(before_dates) == 0 or len(after_dates) == 0:
                continue

            pre_date = before_dates[-1]
            post_date = after_dates[0]

            pre_close = float(hist.loc[pre_date, "Close"])
            post_close = float(hist.loc[post_date, "Close"])
            post_volume = float(hist.loc[post_date, "Volume"])

            avg_vol_window = hist.loc[hist_dates < report_date_naive, "Volume"].tail(20)
            avg_vol = avg_vol_window.mean() if len(avg_vol_window) > 0 else None
            vol_ratio = (post_volume / avg_vol) if avg_vol and avg_vol > 0 else None

            price_reaction_pct = (post_close - pre_close) / pre_close * 100 if pre_close else None

            # Nifty 50's move over the SAME two dates, for comparison
            nifty_reaction_pct = None
            if nifty_hist is not None and pre_date in nifty_dates and post_date in nifty_dates:
                nifty_pre = float(nifty_hist.loc[pre_date, "Close"])
                nifty_post = float(nifty_hist.loc[post_date, "Close"])
                nifty_reaction_pct = (nifty_post - nifty_pre) / nifty_pre * 100 if nifty_pre else None

            excess_reaction_pct = (
                price_reaction_pct - nifty_reaction_pct
                if price_reaction_pct is not None and nifty_reaction_pct is not None
                else None
            )

            eps_estimate = row.get("EPS Estimate")
            reported_eps = row.get("Reported EPS")
            surprise_pct = row.get("Surprise(%)")

            results.append({
                "Report Date": report_date_naive.strftime("%Y-%m-%d"),
                "_report_date_obj": report_date_naive,
                "EPS Estimate": round(float(eps_estimate), 2) if pd.notna(eps_estimate) else None,
                "Reported EPS": round(float(reported_eps), 2) if pd.notna(reported_eps) else None,
                "EPS Surprise %": round(float(surprise_pct), 2) if pd.notna(surprise_pct) else None,
                "Price Reaction %": round(price_reaction_pct, 2) if price_reaction_pct is not None else None,
                "Nifty 50 Reaction %": round(nifty_reaction_pct, 2) if nifty_reaction_pct is not None else None,
                "Excess Reaction %": round(excess_reaction_pct, 2) if excess_reaction_pct is not None else None,
                "Volume vs 20d Avg": round(vol_ratio, 2) if vol_ratio is not None else None,
            })
        except Exception:
            continue  # skip this quarter if data alignment fails, rather than failing the whole result

    if not results:
        return None, "Could not compute price/volume reaction for any past earnings date."

    return results, None


def find_earnings_anomalies(results, surprise_threshold=5.0, reaction_threshold=2.0):
    """
    Flags quarters where the stock's EXCESS reaction (vs Nifty 50, isolating
    stock-specific movement) went the OPPOSITE direction from what the EPS
    surprise would suggest - e.g. a solid earnings beat but the stock still
    underperformed the market, or a miss that still outperformed the market.
    This is a factual flag for further research, not a signal to act on.
    """
    anomalies = []
    for r in results:
        surprise = r.get("EPS Surprise %")
        excess = r.get("Excess Reaction %")
        if surprise is None or excess is None:
            continue
        if surprise >= surprise_threshold and excess <= -reaction_threshold:
            anomalies.append(r)
        elif surprise <= -surprise_threshold and excess >= reaction_threshold:
            anomalies.append(r)
    return anomalies


def research_anomaly_reason(symbol, company_name, report_date_str, surprise_pct, excess_reaction_pct):
    """
    Uses the Claude API (with web search) to research factual, disclosed reasons
    for a specific earnings-reaction anomaly. Requires ANTHROPIC_API_KEY to be
    set. Returns (summary_text, error_message). This is opt-in and makes a real,
    billed API call each time it's used - only call this for specific flagged
    anomalies, not in bulk.

    IMPORTANT: this returns FACTUAL findings from news search, not a recommendation.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY is not set in your environment."

    try:
        import anthropic
    except ImportError:
        return None, "The 'anthropic' package isn't installed. Run: pip install anthropic"

    direction = "beat estimates but the stock still underperformed the Nifty 50" if surprise_pct > 0 else \
                "missed estimates but the stock still outperformed the Nifty 50"

    prompt = (
        f"On or around {report_date_str}, {company_name} ({symbol}.NS) reported quarterly results "
        f"that {direction} (EPS surprise: {surprise_pct}%, excess price reaction vs Nifty 50: {excess_reaction_pct}%). "
        f"Search for news from around that date and summarize the FACTUAL, disclosed reasons reported "
        f"at the time for this reaction - e.g. management guidance, analyst commentary, sector-wide moves, "
        f"promoter actions, regulatory news. Keep it to 3-4 sentences, cite what was actually reported, "
        f"and do not offer any investment recommendation or opinion on whether this was a good or bad sign."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        )
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        summary = "\n".join(text_parts).strip()
        if not summary:
            return None, "No summary was returned - the search may not have found relevant results."
        return summary, None
    except Exception as e:
        return None, f"API call failed: {e}"


def compare_latest_to_history(earnings_results):
    """
    Purely factual comparison of the most recent quarter against the stock's own
    historical pattern (surprise magnitude, excess reaction magnitude, and whether
    it matches a recurring anomaly pattern). Deliberately contains no directive
    language (buy/sell/should) - describes the pattern, doesn't act on it.
    """
    if len(earnings_results) < 2:
        return None, "Need at least 2 quarters of data (1 latest + history to compare against)."

    latest = earnings_results[0]
    historical = earnings_results[1:]

    lines = []

    lines.append(
        f"**Latest report ({latest['Report Date']}):** EPS Surprise {latest['EPS Surprise %']}%, "
        f"Price Reaction {latest['Price Reaction %']}%, Nifty 50 Reaction {latest['Nifty 50 Reaction %']}%, "
        f"Excess Reaction {latest['Excess Reaction %']}%."
    )

    hist_surprises = [r["EPS Surprise %"] for r in historical if r["EPS Surprise %"] is not None]
    if hist_surprises and latest["EPS Surprise %"] is not None:
        avg_surprise = sum(hist_surprises) / len(hist_surprises)
        lines.append(
            f"This quarter's EPS surprise ({latest['EPS Surprise %']}%) compares to a historical "
            f"average of {avg_surprise:.2f}% across the {len(hist_surprises)} prior quarters shown."
        )

    hist_excess = [r["Excess Reaction %"] for r in historical if r["Excess Reaction %"] is not None]
    if hist_excess and latest["Excess Reaction %"] is not None:
        avg_excess = sum(hist_excess) / len(hist_excess)
        lines.append(
            f"This quarter's excess reaction vs Nifty 50 ({latest['Excess Reaction %']}%) compares "
            f"to a historical average of {avg_excess:.2f}%."
        )

    latest_anomaly = find_earnings_anomalies([latest])
    if latest_anomaly:
        direction = ("beat estimates but reacted worse than the Nifty 50"
                     if latest["EPS Surprise %"] > 0
                     else "missed estimates but reacted better than the Nifty 50")
        lines.append(f"This quarter itself is flagged as an anomaly: it {direction}.")
    else:
        lines.append("This quarter's reaction direction was broadly consistent with its EPS surprise direction (not flagged as an anomaly).")

    hist_anomalies = find_earnings_anomalies(historical)
    if hist_anomalies:
        beat_underperform = sum(1 for a in hist_anomalies if a["EPS Surprise %"] > 0)
        miss_outperform = sum(1 for a in hist_anomalies if a["EPS Surprise %"] < 0)
        lines.append(
            f"Historically, {len(hist_anomalies)} of the {len(historical)} prior quarters shown were "
            f"anomalies ({beat_underperform} beat-but-underperformed, {miss_outperform} missed-but-outperformed)."
        )
        if latest_anomaly and latest["EPS Surprise %"] > 0 and beat_underperform > miss_outperform:
            lines.append("This matches a recurring pattern in this stock's history: beats have often still led to underperformance vs the Nifty 50.")
        elif latest_anomaly and latest["EPS Surprise %"] < 0 and miss_outperform > beat_underperform:
            lines.append("This matches a recurring pattern in this stock's history: misses have often still led to outperformance vs the Nifty 50.")
    else:
        lines.append(f"No anomalies were found in the {len(historical)} prior quarters shown - this stock's reaction has historically moved in the expected direction relative to its EPS surprise.")

    return "\n\n".join(lines), None


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

tab1, tab2, tab3, tab4 = st.tabs(["Pair Spread Support/Resistance", "Support & Resistance Finder", "Unusual Volume Scanner", "Quarterly Earnings Reaction"])

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

                dates_present = sorted(set(r["Date"] for r in results_1min))
                latest_date = dates_present[-1]

                current_day_results = [r for r in results_1min if r["Date"] == latest_date]
                previous_results = [r for r in results_1min if r["Date"] != latest_date]

                st.subheader(f"Latest Session ({latest_date})")
                if current_day_results:
                    st.table(pd.DataFrame(current_day_results))
                else:
                    st.info(f"No unusual volume flagged on {latest_date}.")

                if previous_results:
                    st.subheader("Previous Sessions")

                    by_date = {}
                    for r in previous_results:
                        by_date.setdefault(r["Date"], []).append(r)

                    for date in sorted(by_date.keys(), reverse=True):
                        entries = by_date[date]
                        max_ratio_by_symbol = {}
                        for e in entries:
                            max_ratio_by_symbol[e["Symbol"]] = max(max_ratio_by_symbol.get(e["Symbol"], 0), e["Volume Ratio"])
                        sorted_symbols = sorted(max_ratio_by_symbol.items(), key=lambda x: -x[1])
                        symbol_desc = ", ".join(f"{sym} ({ratio}x)" for sym, ratio in sorted_symbols)
                        st.write(f"**{date}:** Unusual volume seen in {symbol_desc}.")
            else:
                st.info("No stocks matched the unusual volume threshold with these settings.")

# ============================================================
# TAB 4: QUARTERLY EARNINGS REACTION
# ============================================================

with tab4:
    st.write("See how a stock has historically reacted (price and volume) to its last several quarterly earnings reports, compared against the Nifty 50's move on the same day.")
    st.caption(
        "This shows FACTUAL data only - EPS estimate vs. actual, the stock's real price/volume "
        "reaction, and how that compares to the Nifty 50's move the same day (\"Excess Reaction\"). "
        "It does not produce a buy/sell recommendation - that judgment is yours to make. Analyst "
        "estimate coverage on Yahoo Finance also tends to be sparser for Indian mid/small-cap "
        "stocks - some fields may show blank if no estimate data is available for a given quarter."
    )

    earnings_symbol_display = st.selectbox(
        "Select stock", options=display_options,
        index=default_index(display_options, "RELIANCE"), key="earnings_symbol"
    )
    num_quarters = st.slider("Number of past quarters to show", min_value=2, max_value=40, value=8, key="earnings_num_quarters")

    if st.button("Get Earnings Reaction History", key="run_earnings_reaction"):
        earnings_symbol = extract_symbol(earnings_symbol_display)
        company_name_row = symbol_df[symbol_df["SYMBOL"] == earnings_symbol]
        company_name = company_name_row["NAME OF COMPANY"].iloc[0] if not company_name_row.empty else earnings_symbol

        bulk_df = load_bulk_earnings_csv()
        nifty_hist, nifty_dates = load_nifty_long_history()

        earnings_results = None
        error = None
        data_source_label = None

        if bulk_df is not None:
            with st.spinner(f"Loading {earnings_symbol} from your downloaded bulk data..."):
                earnings_results = get_earnings_reaction_from_bulk(
                    earnings_symbol, bulk_df, nifty_hist, nifty_dates, num_quarters
                )
            if earnings_results:
                data_source_label = "bulk download (quarterly_earnings_reports_top500.csv)"

        if not earnings_results:
            with st.spinner(f"Not found in bulk data - fetching {earnings_symbol} live from yfinance..."):
                earnings_results, error = get_earnings_reaction(earnings_symbol, num_quarters=num_quarters)
            if earnings_results:
                data_source_label = "live yfinance fetch"

        if error and not earnings_results:
            st.warning(error)
        elif earnings_results:
            st.session_state["earnings_results"] = earnings_results
            st.session_state["earnings_symbol_current"] = earnings_symbol
            st.session_state["earnings_company_name"] = company_name
            st.session_state["earnings_source_label"] = data_source_label

    if "earnings_results" in st.session_state:
        earnings_results = st.session_state["earnings_results"]
        earnings_symbol = st.session_state["earnings_symbol_current"]
        company_name = st.session_state["earnings_company_name"]

        st.subheader(f"{earnings_symbol}: Last {len(earnings_results)} Quarterly Reports")
        if "earnings_source_label" in st.session_state:
            st.caption(f"Source: {st.session_state['earnings_source_label']}")

        display_df = pd.DataFrame(earnings_results).drop(columns=["_report_date_obj"], errors="ignore")
        st.table(display_df)

        avg_excess = display_df["Excess Reaction %"].dropna().mean() if "Excess Reaction %" in display_df else None
        if pd.notna(avg_excess):
            st.metric("Average Excess Reaction vs Nifty 50 (all shown quarters)", f"{avg_excess:.2f}%")

        # --- Anomaly detection: cases where excess reaction contradicts the EPS surprise direction ---
        anomalies = find_earnings_anomalies(earnings_results)

        if anomalies:
            st.subheader("Flagged Anomalies")
            st.caption(
                "Quarters where the stock's reaction (vs Nifty 50) went the OPPOSITE direction from "
                "what the EPS surprise would suggest - e.g. a clear beat that still underperformed "
                "the market, or a miss that still outperformed it. Flagged for further research, not a signal."
            )

            for a in anomalies:
                st.write(
                    f"**{a['Report Date']}:** EPS Surprise {a['EPS Surprise %']}%, "
                    f"Excess Reaction vs Nifty {a['Excess Reaction %']}%"
                )

                research_key = f"research_{earnings_symbol}_{a['Report Date']}"
                if st.button(f"Research why ({a['Report Date']})", key=f"btn_{research_key}"):
                    with st.spinner("Searching for factual context via Claude API (this makes a billed API call)..."):
                        summary, research_error = research_anomaly_reason(
                            earnings_symbol, company_name, a["Report Date"],
                            a["EPS Surprise %"], a["Excess Reaction %"]
                        )
                    if research_error:
                        st.error(research_error)
                    else:
                        st.info(summary)
        else:
            st.caption("No anomalies flagged (stock's reaction direction was broadly consistent with its EPS surprise direction, relative to the Nifty 50).")

        # --- Factual comparison of latest quarter vs this stock's own history ---
        st.subheader("Compare Latest Report to History")
        st.caption(
            "A factual comparison only - how the most recent quarter's numbers compare to this "
            "stock's own historical pattern. This does not produce a buy/sell lean or recommendation."
        )
        if st.button("Compare Latest Report", key="run_compare_latest"):
            comparison_text, compare_error = compare_latest_to_history(earnings_results)
            if compare_error:
                st.warning(compare_error)
            else:
                st.write(comparison_text)