"""
黃金 & 白銀 轉折信號系統 — Streamlit 版本
==========================================
免費部署於 Streamlit Cloud，電腦/手機皆可使用。
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  頁面設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="金銀轉折信號",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 深色主題 CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f !important;
    color: #e8eaf0;
  }
  [data-testid="stSidebar"] { background-color: #0f0f18; }
  [data-testid="metric-container"] {
    background: #141420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 16px;
  }
  div[data-testid="stMetricValue"] { font-family: 'Share Tech Mono', monospace; font-size: 1.6rem; }
  div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
  h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 2px; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; }
  .stAlert { border-radius: 12px; }
  div[data-testid="column"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  常數
# ─────────────────────────────────────────
SYMBOLS = {"🥇 黃金現貨": "GC=F", "🥈 白銀現貨": "SI=F"}
LOOKBACK = 500   # 交易日
REVERSAL_WINDOW = 5
SIGNAL_THRESHOLD = 0.58

FEATURE_COLS = [
    'RSI_14','RSI_7','MACD','MACD_Hist','MACD_Signal',
    'BB_Pct','BB_Width','Stoch_K','Stoch_D',
    'CCI_20','WilliamsR','ADX','ADX_pos','ADX_neg',
    'ZScore_20','ZScore_60','Price_vs_EMA20','Price_vs_EMA50',
    'Return_1d','Return_5d','Return_20d','Volatility_20d','ATR_14',
]

# ─────────────────────────────────────────
#  資料下載（快取 15 分鐘）
# ─────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def download(symbol: str, days: int = LOOKBACK) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ─────────────────────────────────────────
#  技術指標
# ─────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['EMA_20']  = ta.trend.ema_indicator(c, 20)
    df['EMA_50']  = ta.trend.ema_indicator(c, 50)
    df['EMA_200'] = ta.trend.ema_indicator(c, 200)
    macd = ta.trend.MACD(c)
    df['MACD'] = macd.macd(); df['MACD_Signal'] = macd.macd_signal(); df['MACD_Hist'] = macd.macd_diff()
    df['RSI_14'] = ta.momentum.rsi(c, 14); df['RSI_7'] = ta.momentum.rsi(c, 7)
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df['BB_Upper'] = bb.bollinger_hband(); df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg(); df['BB_Pct'] = bb.bollinger_pband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['ATR_14'] = ta.volatility.average_true_range(h, l, c, 14)
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df['Stoch_K'] = stoch.stoch(); df['Stoch_D'] = stoch.stoch_signal()
    df['CCI_20'] = ta.trend.cci(h, l, c, 20)
    df['WilliamsR'] = ta.momentum.williams_r(h, l, c, 14)
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df['ADX'] = adx.adx(); df['ADX_pos'] = adx.adx_pos(); df['ADX_neg'] = adx.adx_neg()
    df['OBV'] = ta.volume.on_balance_volume(c, v)
    df['Return_1d']  = c.pct_change(1)
    df['Return_5d']  = c.pct_change(5)
    df['Return_20d'] = c.pct_change(20)
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)
    df['ZScore_20'] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df['ZScore_60'] = (c - c.rolling(60).mean()) / c.rolling(60).std()
    df['Price_vs_EMA20'] = (c - df['EMA_20']) / df['EMA_20']
    df['Price_vs_EMA50'] = (c - df['EMA_50']) / df['EMA_50']
    return df

# ─────────────────────────────────────────
#  標記轉折
# ─────────────────────────────────────────
def label_reversals(df: pd.DataFrame, w: int = REVERSAL_WINDOW) -> pd.DataFrame:
    c = df['Close'].values
    atr = df['ATR_14'].values
    labels = np.zeros(len(df))
    for i in range(w, len(df) - w):
        if c[i] <= c[i-w:i].min() and (c[i:i+w].max() - c[i]) > atr[i]:
            labels[i] = 1
        elif c[i] >= c[i-w:i].max() and (c[i] - c[i:i+w].min()) > atr[i]:
            labels[i] = -1
    df['Label'] = labels
    return df

# ─────────────────────────────────────────
#  ML 訓練與預測
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def train_and_predict(df_json: str):
    df = pd.read_json(df_json)
    df = label_reversals(df)
    clean = df.dropna(subset=FEATURE_COLS + ['Label'])
    X = clean[FEATURE_COLS].values
    y = clean['Label'].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf  = RandomForestClassifier(200, max_depth=6, class_weight='balanced', random_state=42)
    gbt = GradientBoostingClassifier(150, max_depth=4, learning_rate=0.05, random_state=42)
    rf.fit(Xs, y); gbt.fit(Xs, y)

    all_clean = df.dropna(subset=FEATURE_COLS)
    Xall = scaler.transform(all_clean[FEATURE_COLS].values)
    rp = rf.predict_proba(Xall); gp = gbt.predict_proba(Xall)
    avg = (rp + gp) / 2
    classes = list(rf.classes_)
    bot_idx = classes.index(1)  if 1  in classes else None
    top_idx = classes.index(-1) if -1 in classes else None

    df.loc[all_clean.index, 'ML_Bot'] = avg[:, bot_idx] if bot_idx is not None else 0
    df.loc[all_clean.index, 'ML_Top'] = avg[:, top_idx] if top_idx is not None else 0
    return df

# ─────────────────────────────────────────
#  規則信號 + 綜合
# ─────────────────────────────────────────
def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    bot = (
        (df['RSI_14'] < 35) & (df['Stoch_K'] < 25) &
        (df['BB_Pct'] < 0.15) & (df['MACD_Hist'] > df['MACD_Hist'].shift(1))
    )
    top = (
        (df['RSI_14'] > 65) & (df['Stoch_K'] > 75) &
        (df['BB_Pct'] > 0.85) & (df['MACD_Hist'] < df['MACD_Hist'].shift(1))
    )
    z_bot = df['ZScore_20'] < -2.0
    z_top = df['ZScore_20'] >  2.0
    rule = pd.Series(0, index=df.index)
    rule[bot | z_bot] += 1; rule[top | z_top] -= 1
    df['Rule'] = rule.clip(-1, 1)

    ml_sig = pd.Series(0, index=df.index)
    if 'ML_Bot' in df.columns:
        ml_sig[df['ML_Bot'] >= SIGNAL_THRESHOLD] = 1
        ml_sig[df['ML_Top'] >= SIGNAL_THRESHOLD] = -1
    df['ML_Sig'] = ml_sig

    combined = df['Rule'] + df['ML_Sig']
    df['Signal'] = 0
    df.loc[combined >= 2,  'Signal'] = 2   # 強底
    df.loc[combined == 1,  'Signal'] = 1   # 弱底
    df.loc[combined <= -2, 'Signal'] = -2  # 強頂
    df.loc[combined == -1, 'Signal'] = -1  # 弱頂
    return df

# ─────────────────────────────────────────
#  新聞 RSS（快取 20 分鐘）
# ─────────────────────────────────────────
@st.cache_data(ttl=1200, show_spinner=False)
def fetch_news() -> list[dict]:
    feeds = [
        ("Reuters 商品", "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters 市場", "https://feeds.reuters.com/reuters/marketsNews"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ]
    items = []
    gold_kw = ['gold','silver','metal','precious','commodity','Fed','inflation',
               '黃金','白銀','貴金屬','聯準會','通膨','利率']
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=6,
                headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(r.content)
            for item in root.iter('item'):
                title = item.findtext('title', '')
                link  = item.findtext('link', '')
                pub   = item.findtext('pubDate', '')
                if any(k.lower() in title.lower() for k in gold_kw):
                    items.append({'source': src, 'title': title,
                                  'link': link, 'pub': pub})
        except Exception:
            pass
    return items[:20]

# ─────────────────────────────────────────
#  市場環境資料（快取 30 分鐘）
# ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro() -> dict:
    tickers = {
        'DXY': 'DX-Y.NYB',   # 美元指數
        'TLT': 'TLT',         # 長債 (利率代理)
        'VIX': '^VIX',        # 恐慌指數
        'SPY': 'SPY',         # 股市
        'TIP': 'TIP',         # 抗通膨債 (實質利率代理)
        'OIL': 'CL=F',        # 原油
    }
    result = {}
    for name, sym in tickers.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period='5d')
            if len(hist) >= 2:
                cur = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                result[name] = {'value': cur, 'chg': (cur - prev) / prev * 100}
        except Exception:
            result[name] = {'value': None, 'chg': None}
    return result

# ─────────────────────────────────────────
#  COT 簡易解讀（CFTC 公開資料）
# ─────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cot_summary() -> str:
    """CFTC COT 報告每週二更新，這裡提供說明與連結"""
    return (
        "COT 報告每週五發布（反映週二倉位）。\n"
        "• 大型投機者（Managed Money）淨多單增加 → 看漲情緒升溫\n"
        "• 商業避險者（Commercial）淨空單擴大 → 反向指標偏多\n"
        "• 極端淨多/淨空 → 常見轉折前兆\n\n"
        "🔗 最新 COT 資料：https://www.cftc.gov/dea/futures/deacmesf.htm"
    )

# ─────────────────────────────────────────
#  環境影響判斷
# ─────────────────────────────────────────
def macro_interpretation(macro: dict) -> list[dict]:
    signals = []
    dxy = macro.get('DXY', {})
    vix = macro.get('VIX', {})
    tlt = macro.get('TLT', {})
    tip = macro.get('TIP', {})
    oil = macro.get('OIL', {})

    if dxy.get('chg') is not None:
        if dxy['chg'] > 0.3:
            signals.append({'icon':'🔴','factor':'美元指數','msg':f"上漲 {dxy['chg']:.2f}%，對黃金形成壓力",'bias':'空'})
        elif dxy['chg'] < -0.3:
            signals.append({'icon':'🟢','factor':'美元指數','msg':f"下跌 {abs(dxy['chg']):.2f}%，利多黃金白銀",'bias':'多'})
        else:
            signals.append({'icon':'⚪','factor':'美元指數','msg':f"變動 {dxy['chg']:.2f}%，影響中性",'bias':'中性'})

    if vix.get('value') is not None:
        v = vix['value']
        if v > 25:
            signals.append({'icon':'🟡','factor':'VIX 恐慌指數','msg':f"VIX={v:.1f}，市場恐慌，黃金避險需求上升",'bias':'多'})
        elif v < 15:
            signals.append({'icon':'⚪','factor':'VIX 恐慌指數','msg':f"VIX={v:.1f}，市場平靜，避險需求低",'bias':'中性'})
        else:
            signals.append({'icon':'⚪','factor':'VIX 恐慌指數','msg':f"VIX={v:.1f}，正常範圍",'bias':'中性'})

    if tip.get('chg') is not None:
        t = tip['chg']
        if t < -0.3:
            signals.append({'icon':'🟢','factor':'實質利率(TIP)','msg':f"TIP 下跌 {abs(t):.2f}%，實質利率走低，利多黃金",'bias':'多'})
        elif t > 0.3:
            signals.append({'icon':'🔴','factor':'實質利率(TIP)','msg':f"TIP 上漲 {t:.2f}%，實質利率走高，壓制黃金",'bias':'空'})
        else:
            signals.append({'icon':'⚪','factor':'實質利率(TIP)','msg':f"TIP 變動 {t:.2f}%，影響中性",'bias':'中性'})

    if oil.get('chg') is not None:
        o = oil['chg']
        if o > 2:
            signals.append({'icon':'🟡','factor':'原油','msg':f"原油上漲 {o:.2f}%，通膨預期升溫，間接利多金銀",'bias':'偏多'})
        elif o < -2:
            signals.append({'icon':'⚪','factor':'原油','msg':f"原油下跌 {abs(o):.2f}%，通膨預期降溫",'bias':'中性'})

    return signals

# ─────────────────────────────────────────
#  信號評分
# ─────────────────────────────────────────
def score_signal(row: pd.Series) -> tuple[int, str, str]:
    s = int(row.get('Signal', 0))
    rsi = row.get('RSI_14', 50)
    zs  = row.get('ZScore_20', 0)
    ml_b = row.get('ML_Bot', 0)
    ml_t = row.get('ML_Top', 0)

    if s == 2:
        pct = min(100, int(60 + (ml_b * 30) + (max(0, -zs - 2) * 5)))
        return pct, "🟢 強底部信號", "bull"
    elif s == 1:
        pct = min(70, int(40 + ml_b * 20))
        return pct, "🟡 弱底部信號（待確認）", "warn"
    elif s == -2:
        pct = min(100, int(60 + (ml_t * 30) + (max(0, zs - 2) * 5)))
        return pct, "🔴 強頂部信號", "bear"
    elif s == -1:
        pct = min(70, int(40 + ml_t * 20))
        return pct, "🟡 弱頂部信號（待確認）", "warn"
    else:
        return 0, "⚪ 無明確信號", "neutral"

# ─────────────────────────────────────────
#  Plotly 圖表
# ─────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
    font=dict(color='#8899aa', family='Share Tech Mono'),
    xaxis=dict(gridcolor='#1a1a2e', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1a1a2e', showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
)

def plot_price(df: pd.DataFrame, name: str) -> go.Figure:
    recent = df.tail(120)
    color = '#d4a843' if '黃金' in name else '#a8b8c8'

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.03)

    # 蠟燭圖
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent['Open'], high=recent['High'],
        low=recent['Low'], close=recent['Close'],
        increasing_line_color='#00e5a0', decreasing_line_color='#ff4757',
        name='K線', showlegend=False
    ), row=1, col=1)

    for col, lw, dash in [('EMA_20', 1.5, 'solid'), ('EMA_50', 1.2, 'dash'), ('BB_Upper', 0.8, 'dot'), ('BB_Lower', 0.8, 'dot')]:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent[col], name=col,
            line=dict(color=color, width=lw, dash=dash), opacity=0.6
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd.concat([recent.index.to_series(), recent.index.to_series()[::-1]]),
        y=pd.concat([recent['BB_Upper'], recent['BB_Lower'][::-1]]),
        fill='toself', fillcolor=f'rgba(212,168,67,0.05)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='布林帶'
    ), row=1, col=1)

    # 信號標記
    for sig_val, marker, mcolor, label in [
        (2,  'triangle-up',   '#00e5a0', '強底'),
        (1,  'triangle-up',   '#56d364', '弱底'),
        (-2, 'triangle-down', '#ff4757', '強頂'),
        (-1, 'triangle-down', '#ffa198', '弱頂'),
    ]:
        pts = recent[recent['Signal'] == sig_val]
        if len(pts):
            y_val = pts['Low'] * 0.998 if sig_val > 0 else pts['High'] * 1.002
            fig.add_trace(go.Scatter(
                x=pts.index, y=y_val, mode='markers',
                marker=dict(symbol=marker, size=12, color=mcolor),
                name=label
            ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=recent.index, y=recent['RSI_14'],
        line=dict(color='#a371f7', width=1.5), name='RSI'), row=2, col=1)
    for lvl, c in [(70, '#ff4757'), (30, '#00e5a0')]:
        fig.add_hline(y=lvl, line_color=c, line_dash='dash', opacity=0.5, row=2, col=1)

    # MACD
    hist_colors = ['#00e5a0' if v >= 0 else '#ff4757' for v in recent['MACD_Hist']]
    fig.add_trace(go.Bar(x=recent.index, y=recent['MACD_Hist'],
        marker_color=hist_colors, name='MACD Hist', opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD'],
        line=dict(color='#58a6ff', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD_Signal'],
        line=dict(color='#ffa657', width=1), name='Signal'), row=3, col=1)

    fig.update_layout(
        **PLOT_THEME,
        title=dict(text=f'{name} | 近 120 日', font=dict(size=14, color='#c9d1d9')),
        height=520, showlegend=False,
        xaxis3=dict(rangeslider=dict(visible=False)),
    )
    fig.update_yaxes(row=2, col=1, title='RSI', range=[0, 100])
    fig.update_yaxes(row=3, col=1, title='MACD')
    return fig

def plot_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> go.Figure:
    ratio = (gold_df['Close'] / silver_df['Close']).dropna().tail(250)
    ma60  = ratio.rolling(60).mean()
    std60 = ratio.rolling(60).std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
        line=dict(color='#d4a843', width=2), name='金銀比'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma60.index, y=ma60,
        line=dict(color='#a8b8c8', width=1, dash='dash'), name='60日均'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([ma60.index.to_series(), ma60.index.to_series()[::-1]]),
        y=pd.concat([ma60 + 2*std60, (ma60 - 2*std60)[::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.06)',
        line=dict(color='rgba(0,0,0,0)'), name='±2σ'
    ), row=1, col=1)

    zscore = ((ratio - ma60) / std60)
    colors = ['#ff4757' if z > 0 else '#00e5a0' for z in zscore]
    fig.add_trace(go.Bar(x=zscore.index, y=zscore,
        marker_color=colors, opacity=0.7, name='Z-Score'), row=2, col=1)
    for lvl, c in [(2, '#ff4757'), (-2, '#00e5a0')]:
        fig.add_hline(y=lvl, line_color=c, line_dash='dash', opacity=0.5, row=2, col=1)

    fig.update_layout(**PLOT_THEME, title=dict(text='黃金/白銀 比值分析',
        font=dict(size=14, color='#c9d1d9')), height=400)
    return fig

# ─────────────────────────────────────────
#  主介面
# ─────────────────────────────────────────
def main():
    # 標題
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.markdown("# 🏅 金銀轉折信號系統")
    with col_time:
        st.markdown(f"<div style='text-align:right;color:#4a5a6a;font-size:12px;padding-top:20px;font-family:monospace'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)

    # 載入資料
    with st.spinner("載入市場資料中…"):
        dfs = {}
        for name, sym in SYMBOLS.items():
            try:
                raw = download(sym)
                raw = add_indicators(raw)
                raw = train_and_predict(raw.to_json())
                raw = add_signals(raw)
                dfs[name] = raw
            except Exception as e:
                st.warning(f"{name} 資料載入失敗: {e}")

        macro = fetch_macro()

    if not dfs:
        st.error("無法載入任何市場資料，請稍後重試。")
        return

    # ── 即時價格總覽 ──
    st.markdown("### 📊 即時報價")
    metric_cols = st.columns(len(dfs) * 2)
    col_idx = 0
    for name, df in dfs.items():
        if df.empty: continue
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        chg    = latest['Close'] - prev['Close']
        chg_pct = chg / prev['Close'] * 100
        unit = "USD/盎司"
        with metric_cols[col_idx]:
            st.metric(f"{name}", f"{latest['Close']:.2f}",
                      delta=f"{chg:+.2f} ({chg_pct:+.2f}%)")
        col_idx += 1
        # 信號評分
        pct, label, bias = score_signal(latest)
        with metric_cols[col_idx]:
            color_map = {'bull':'#00e5a0','bear':'#ff4757','warn':'#ffbe00','neutral':'#8899aa'}
            c = color_map.get(bias, '#8899aa')
            if pct > 0:
                st.markdown(f"<div style='padding:10px;background:#141420;border-radius:12px;border:1px solid {c}33'>"
                            f"<div style='color:{c};font-size:13px;font-weight:600'>{label}</div>"
                            f"<div style='color:{c};font-size:22px;font-family:monospace'>{pct}分</div>"
                            f"</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:10px;background:#141420;border-radius:12px;border:1px solid #1a1a2e'>"
                            f"<div style='color:#4a5a6a;font-size:13px'>{label}</div>"
                            f"<div style='color:#4a5
                            
