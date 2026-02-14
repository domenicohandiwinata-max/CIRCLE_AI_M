import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================================================
# CONFIG & API SETUP
# ==================================================
ALPHA_VANTAGE_API_KEY = "MASUKKAN_API_KEY_KAMU_DISINI"

st.set_page_config(
    page_title="CIRCLE AI MULTI-MARKET PRO",
    page_icon="🤖",
    layout="wide"
)

# UI Styling Modern
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1C1F26; padding: 15px; border-radius: 10px; border: 1px solid #31333F; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .signal-buy { color: #00FF9C; font-weight: 900; font-size: 2rem; }
    .signal-sell { color: #FF4B4B; font-weight: 900; font-size: 2rem; }
    .signal-wait { color: #FFD700; font-weight: 900; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# CORE AI ENGINE
# ==================================================

def calculate_indicators(df):
    df = df.copy()
    # RSI Calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA7'] = df['close'].rolling(7).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['Volatility'] = df['close'].rolling(14).std()
    
    # Target for ML (1 if price goes up tomorrow)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

def run_ai_prediction(df):
    features = ['close', 'MA7', 'MA30', 'RSI', 'Volatility']
    X = df[features]
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_scaled, y)
    
    last_row = X.iloc[-1:]
    proba = model.predict_proba(scaler.transform(last_row))[0]
    return proba

# ==================================================
# DATA ACQUISITION
# ==================================================

@st.cache_data(ttl=300)
def get_forex_data(base, quote):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": base,
        "to_symbol": quote,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "Time Series FX (Daily)" in data:
        df = pd.DataFrame(data["Time Series FX (Daily)"]).T
        df.columns = ['open', 'high', 'low', 'close']
        df = df.astype(float).sort_index()
        return df, None
    return None, data.get("Note", "Error fetching data")

def get_stock_data(symbol):
    data = yf.download(symbol, period="1y", interval="1d", progress=False)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        return data, None
    return None, "Symbol not found"

# ==================================================
# SIDEBAR CONTROL
# ==================================================
with st.sidebar:
    st.title("🤖 CIRCLE AI")
    mode = st.radio("Pilih Mode Analisis", ["📈 Saham (Yahoo)", "💱 Forex (AlphaV)"])
    st.divider()
    
    if mode == "📈 Saham (Yahoo)":
        ticker = st.text_input("Kode Saham", value="BBCA.JK").upper()
    else:
        c1, c2 = st.columns(2)
        base_c = c1.text_input("Base", value="USD").upper()
        quote_c = c2.text_input("Quote", value="IDR").upper()
        ticker = f"{base_c}/{quote_c}"
        
    st.caption("AI menganalisis tren 365 hari terakhir untuk memberikan sinyal.")

# ==================================================
# MAIN DASHBOARD RENDERING
# ==================================================
st.title(f"Market Intelligence: {ticker}")

# Data Loading Logic
with st.spinner('AI sedang memproses data pasar...'):
    if "Saham" in mode:
        df_raw, err = get_stock_data(ticker)
    else:
        df_raw, err = get_forex_data(base_c, quote_c)

if df_raw is not None:
    df_proc = calculate_indicators(df_raw)
    proba = run_ai_prediction(df_proc)
    
    # Recommendation Logic
    prob_up = proba[1]
    conf = max(proba) * 100
    last_price = df_proc['close'].iloc[-1]
    
    if conf < 58:
        signal = "WAIT"
        css_class = "signal-wait"
    elif prob_up > 0.5:
        signal = "BUY"
        css_class = "signal-buy"
    else:
        signal = "SELL"
        css_class = "signal-sell"

    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h5>Harga Saat Ini</h5><h2 style="color:#4A90E2;">{last_price:,.2f}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h5>AI Confidence</h5><h2>{conf:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h5>Sinyal AI</h5><p class="{css_class}">{signal}</p></div>', unsafe_allow_html=True)

    # --- CHARTS ---
    tab_chart, tab_data = st.tabs(["📊 Technical Chart", "📑 Raw Data"])
    
    with tab_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_proc.index, open=df_proc['open'], high=df_proc['high'],
            low=df_proc['low'], close=df_proc['close'], name="Candlestick"
        ), row=1, col=1)
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=df_proc.index, y=df_proc['MA7'], name="MA 7 (Fast)", line=dict(color='#00FF9C', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_proc.index, y=df_proc['MA30'], name="MA 30 (Slow)", line=dict(color='#FF4B4B', width=1)), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_proc.index, y=df_proc['RSI'], name="RSI", line=dict(color='#FFD700')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.dataframe(df_proc.tail(50).sort_index(ascending=False), use_container_width=True)

else:
    st.error(f"Data tidak dapat dimuat: {err}")
    st.info("Tips: Jika menggunakan Forex, pastikan API Key benar dan tidak melebihi limit harian.")

st.divider()
st.caption("Disclaimer: Analisis AI adalah alat bantu, bukan jaminan keuntungan. Gunakan manajemen risiko yang baik.")