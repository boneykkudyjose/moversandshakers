import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

# ------------------------------
# Configuration
# ------------------------------
Z_THRESH = 2
PERIODS = [30, 60, 90]
APEWISDOM_FILTER = "all-stocks"
APEWISDOM_PAGES = 2               # Fetch up to 200 trending stocks
MAX_TICKERS_TO_PROCESS = 100      # Process top 100 for momentum
TOP_BUY_SELL = 10                 # Number of top buy/sell charts to show

# ------------------------------
# Helper Functions (from your original code)
# ------------------------------
def monthdelta(date, delta):
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m:
        m = 12
    d = min(date.day, [31, 29 if y % 4 == 0 and (not y % 100 == 0 or y % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date.replace(day=d, month=m, year=y)

@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol, start_date, end_date):
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    ticker = yf.Ticker(ticker_symbol, session=session)
    return ticker.history(start=start_date, end=end_date)

def calculate_z_scores(close_prices, periods):
    z_scores_dict = {}
    for period in periods:
        rolling_mean = close_prices.rolling(window=period).mean()
        rolling_std = close_prices.rolling(window=period).std()
        z_scores = (close_prices - rolling_mean) / rolling_std
        z_scores_dict[period] = z_scores
    return z_scores_dict

def get_buy_sell_signals(z_scores_data, threshold):
    for period, z_scores in z_scores_data.items():
        if period == PERIODS[0]:
            buy = z_scores < -threshold
            sell = z_scores > threshold
            return buy, sell
    return None, None

def get_latest_z_score(ticker):
    """Return the most recent 30‑day Z‑score for a ticker."""
    end = datetime.now()
    start = monthdelta(end, -6)   # 6 months of data
    try:
        hist = fetch_data(ticker, start, end)
        if hist.empty or len(hist) < 30:
            return None
        close = hist['Close']
        z_dict = calculate_z_scores(close, [30])
        latest_z = z_dict[30].iloc[-1]
        return latest_z if not np.isnan(latest_z) else None
    except:
        return None

def plot_mini_chart(ticker, df, figsize=(3, 2)):
    """Create a small chart for a ticker (price + buy/sell markers)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df['price'], color='blue', linewidth=1)
    buy_pts = df[df['buy']]
    sell_pts = df[df['sell']]
    if not buy_pts.empty:
        ax.scatter(buy_pts.index, buy_pts['price'], marker='^', color='g', s=20)
    if not sell_pts.empty:
        ax.scatter(sell_pts.index, sell_pts['price'], marker='v', color='r', s=20)
    # Show threshold lines on mini chart? Optional – makes it busy
    # ax.axhline(y=..., linestyle='--', alpha=0.5)
    ax.set_title(ticker, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Helper for the big plot (reused from your original code)
def plot_ticker_with_signals(ticker, df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['price'], label='Price', color='blue')
    buy_signals = df[df['buy']]
    sell_signals = df[df['sell']]
    ax.scatter(buy_signals.index, buy_signals['price'], marker='^', color='g', label='Buy Signal', s=100)
    ax.scatter(sell_signals.index, sell_signals['price'], marker='v', color='r', label='Sell Signal', s=100)
    ax.set_title(f'{ticker} – Price with Buy/Sell Signals (Z‑score threshold = {Z_THRESH})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig
# ------------------------------
# ApeWisdom API Integration
# ------------------------------
@st.cache_data(ttl=3600)
def fetch_apewisdom_tickers(filter_type="all-stocks", max_pages=2):
    ticker_data = []
    base_url = f"https://apewisdom.io/api/v1.0/filter/{filter_type}"
    for page in range(1, max_pages + 1):
        url = f"{base_url}/page/{page}"
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                for item in results:
                    ticker_data.append({
                        'rank': item.get('rank'),
                        'ticker': item.get('ticker'),
                        'name': item.get('name'),
                        'mentions': item.get('mentions'),
                        'upvotes': item.get('upvotes')
                    })
                if len(results) < 100:
                    break
            else:
                break
        except:
            break
    df = pd.DataFrame(ticker_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def compute_momentum(ticker, end_date, lookback_days=30):
    start_date = end_date - timedelta(days=lookback_days)
    try:
        hist = fetch_data(ticker, start_date, end_date)
        if hist.empty or len(hist) < 2:
            return None
        current = hist['Close'].iloc[-1]
        past = hist['Close'].iloc[0]
        return (current - past) / past * 100
    except:
        return None

def get_stock_info(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        price = info.get('currentPrice', info.get('regularMarketPrice', None))
        volume = info.get('volume', None)
        return price, volume
    except:
        return None, None

# ------------------------------
# Main App
# ------------------------------
st.set_page_config(page_title="ApeWisdom Momentum + Buy/Sell Signals", layout="wide")
st.title("📈 Top 50 Momentum Stocks + Top 10 Buy / Top 10 Sell Charts")

# 1. Fetch trending stocks
with st.spinner("Fetching trending stocks from ApeWisdom..."):
    trending_df = fetch_apewisdom_tickers(APEWISDOM_FILTER, APEWISDOM_PAGES)

if trending_df.empty:
    st.error("Could not retrieve data from ApeWisdom API.")
    st.stop()

st.success(f"Retrieved {len(trending_df)} unique trending stocks")

# 2. Compute momentum for top N
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
tickers_to_process = trending_df['ticker'].tolist()[:MAX_TICKERS_TO_PROCESS]

momentum_data = []
progress_bar = st.progress(0)

for i, ticker in enumerate(tickers_to_process):
    momentum = compute_momentum(ticker, end_date, lookback_days=30)
    if momentum is not None:
        price, volume = get_stock_info(ticker)
        row = trending_df[trending_df['ticker'] == ticker].iloc[0]
        momentum_data.append({
            "Ticker": ticker,
            "Momentum (30d, %)": round(momentum, 2),
            "Current Price ($)": price if price else "N/A",
            "Volume": volume if volume else "N/A",
            "Mentions": row.get('mentions', 'N/A'),
            "Upvotes": row.get('upvotes', 'N/A')
        })
    progress_bar.progress((i + 1) / len(tickers_to_process))

if not momentum_data:
    st.error("No momentum data computed.")
    st.stop()

df_momentum = pd.DataFrame(momentum_data)
df_momentum = df_momentum.sort_values("Momentum (30d, %)", ascending=False).head(50)
st.subheader("🏆 Top 50 Momentum Stocks")
st.dataframe(df_momentum, use_container_width=True)

# ------------------------------
# 3. Compute latest Z‑score for each ticker in top 50
# ------------------------------
st.markdown("---")
st.subheader(f"🔍 Top {TOP_BUY_SELL} Buy Candidates (Most Oversold) & Top {TOP_BUY_SELL} Sell Candidates (Most Overbought)")

tickers_in_top50 = df_momentum['Ticker'].tolist()
z_scores_dict = {}
for ticker in tickers_in_top50:
    z = get_latest_z_score(ticker)
    if z is not None:
        z_scores_dict[ticker] = z

if not z_scores_dict:
    st.warning("Could not compute Z‑scores for any ticker. Skipping charts.")
else:
    # Sort for buy (lowest Z = most oversold) and sell (highest Z)
    buy_candidates = sorted(z_scores_dict.items(), key=lambda x: x[1])[:TOP_BUY_SELL]
    sell_candidates = sorted(z_scores_dict.items(), key=lambda x: -x[1])[:TOP_BUY_SELL]

    # Function to generate mini charts for a list of (ticker, z_score)
    def show_mini_charts(candidates, title, color_hint):
        st.write(f"**{title}**")
        cols = st.columns(5)
        for idx, (ticker, z_score) in enumerate(candidates):
            # Fetch full 6‑month history for plotting
            end = datetime.now()
            start = monthdelta(end, -6)
            hist = fetch_data(ticker, start, end)
            if not hist.empty:
                close = hist['Close']
                z_dict = calculate_z_scores(close, [30])
                buy_sig, sell_sig = get_buy_sell_signals(z_dict, Z_THRESH)
                plot_df = pd.DataFrame({
                    'price': close,
                    'buy': buy_sig if buy_sig is not None else False,
                    'sell': sell_sig if sell_sig is not None else False
                })
                fig = plot_mini_chart(ticker, plot_df)
                with cols[idx % 5]:
                    st.pyplot(fig)
                    st.caption(f"{ticker} | Z = {z_score:.2f}")
            else:
                with cols[idx % 5]:
                    st.write(f"{ticker}: no data")

    show_mini_charts(buy_candidates, f"Top {TOP_BUY_SELL} BUY (lowest Z‑score)", "green")
    st.markdown("---")
    show_mini_charts(sell_candidates, f"Top {TOP_BUY_SELL} SELL (highest Z‑score)", "red")

# ------------------------------
# 4. Optional: Individual ticker deep dive
# ------------------------------
st.markdown("---")
st.subheader("🔍 Detailed Analysis for Any Ticker")
selected_ticker = st.selectbox("Choose a ticker for full‑size chart", tickers_in_top50)

if selected_ticker:
    end = datetime.now()
    start = monthdelta(end, -6)
    with st.spinner(f"Loading data for {selected_ticker}..."):
        hist = fetch_data(selected_ticker, start, end)
        if not hist.empty:
            close = hist['Close']
            z_dict = calculate_z_scores(close, PERIODS)
            buy_sig, sell_sig = get_buy_sell_signals(z_dict, Z_THRESH)
            signal_df = pd.DataFrame({'price': close, 'buy': buy_sig, 'sell': sell_sig})
            fig = plot_ticker_with_signals(selected_ticker, signal_df)   # reuse your original big plot
            st.pyplot(fig)

            # Show Z‑score curves
            fig_z, ax = plt.subplots(figsize=(10, 4))
            for period, zs in z_dict.items():
                ax.plot(zs.index, zs, label=f"{period} days", alpha=0.7)
            ax.axhline(-Z_THRESH, color='r', linestyle='--')
            ax.axhline(Z_THRESH, color='r', linestyle='--')
            ax.set_ylabel("Z‑score")
            ax.legend()
            st.pyplot(fig_z)
        else:
            st.warning(f"No data for {selected_ticker}")
