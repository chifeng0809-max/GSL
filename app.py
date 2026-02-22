import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="金銀轉折信號", page_icon="🥇", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #0a0a0f !important; color: #e8eaf0; }
[data-testid="metric-container"] { background: #141420; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px; }
div[data-testid="stMetricValue"] { font-family: monospace; font-size: 1.6rem; }
h1, h2, h3 { font-family: Rajdhani, sans-serif !important; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

SYMBOLS = {"🥇 黃金現貨": "GC=F", "🥈 白銀現貨": "SI=F"}
FEATURE_COLS = [
    'RSI_14','RSI_7','MACD','MACD_Hist','MACD_Signal',
    'BB_Pct','BB_Width','Stoch_K','Stoch_D',
    'CCI_20','WilliamsR','ADX','ADX_pos','ADX_neg',
    'ZScore_20','ZScore_60','Price_vs_EMA20','Price_vs_EMA50',
    'Return_1d','Return_5d','Return_20d','Volatility_20d','ATR_14',
]

@st.cache_data(ttl=900, show_spinner=False)
def download(symbol):
    end = datetime.today()
    start = end - timedelta(days=750)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def add_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['EMA_20'] = ta.trend.ema_indicator(c, 20)
    df['EMA_50'] = ta.trend.ema_indicator(c, 50)
    df['EMA_200'] = ta.trend.ema_indicator(c, 200)
    macd = ta.trend.MACD(c)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['RSI_14'] = ta.momentum.rsi(c, 14)
    df['RSI_7'] = ta.momentum.rsi(c, 7)
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Pct'] = bb.bollinger_pband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['ATR_14'] = ta.volatility.average_true_range(h, l, c, 14)
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df['CCI_20'] = ta.trend.cci(h, l, c, 20)
    df['WilliamsR'] = ta.momentum.williams_r(h, l, c, 14)
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()
    df['OBV'] = ta.volume.on_balance_volume(c, v)
    df['Return_1d'] = c.pct_change(1)
    df['Return_5d'] = c.pct_change(5)
    df['Return_20d'] = c.pct_change(20)
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)
    df['ZScore_20'] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df['ZScore_60'] = (c - c.rolling(60).mean()) / c.rolling(60).std()
    df['Price_vs_EMA20'] = (c - df['EMA_20']) / df['EMA_20']
    df['Price_vs_EMA50'] = (c - df['EMA_50']) / df['EMA_50']
    return df

def label_reversals(df, w=5):
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

@st.cache_data(ttl=3600, show_spinner=False)
def train_and_predict(df_json):
    df = pd.read_json(df_json)
    df = label_reversals(df)
    clean = df.dropna(subset=FEATURE_COLS + ['Label'])
    X = clean[FEATURE_COLS].values
    y = clean['Label'].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(200, max_depth=6, class_weight='balanced', random_state=42)
    gbt = GradientBoostingClassifier(150, max_depth=4, learning_rate=0.05, random_state=42)
    rf.fit(Xs, y)
    gbt.fit(Xs, y)
    all_clean = df.dropna(subset=FEATURE_COLS)
    Xall = scaler.transform(all_clean[FEATURE_COLS].values)
    rp = rf.predict_proba(Xall)
    gp = gbt.predict_proba(Xall)
    avg = (rp + gp) / 2
    classes = list(rf.classes_)
    if 1 in classes:
        df.loc[all_clean.index, 'ML_Bot'] = avg[:, classes.index(1)]
    else:
        df.loc[all_clean.index, 'ML_Bot'] = 0
    if -1 in classes:
        df.loc[all_clean.index, 'ML_Top'] = avg[:, classes.index(-1)]
    else:
        df.loc[all_clean.index, 'ML_Top'] = 0
    return df

def add_signals(df):
    bot = (
        (df['RSI_14'] < 35) & (df['Stoch_K'] < 25) &
        (df['BB_Pct'] < 0.15) & (df['MACD_Hist'] > df['MACD_Hist'].shift(1))
    )
    top = (
        (df['RSI_14'] > 65) & (df['Stoch_K'] > 75) &
        (df['BB_Pct'] > 0.85) & (df['MACD_Hist'] < df['MACD_Hist'].shift(1))
    )
    rule = pd.Series(0, index=df.index)
    rule[bot | (df['ZScore_20'] < -2)] += 1
    rule[top | (df['ZScore_20'] > 2)] -= 1
    df['Rule'] = rule.clip(-1, 1)
    ml_sig = pd.Series(0, index=df.index)
    if 'ML_Bot' in df.columns:
        ml_sig[df['ML_Bot'] >= 0.58] = 1
        ml_sig[df['ML_Top'] >= 0.58] = -1
    df['ML_Sig'] = ml_sig
    combined = df['Rule'] + df['ML_Sig']
    df['Signal'] = 0
    df.loc[combined >= 2, 'Signal'] = 2
    df.loc[combined == 1, 'Signal'] = 1
    df.loc[combined <= -2, 'Signal'] = -2
    df.loc[combined == -1, 'Signal'] = -1
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro():
    tickers = {'DXY':'DX-Y.NYB','VIX':'^VIX','TIP':'TIP','OIL':'CL=F','SPY':'SPY','TLT':'TLT'}
    result = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period='5d')
            if len(hist) >= 2:
                cur = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                result[name] = {'value': float(cur), 'chg': float((cur-prev)/prev*100)}
        except Exception:
            result[name] = {'value': None, 'chg': None}
    return result

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_news():
    feeds = [
        ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ]
    items = []
    kw = ['gold','silver','metal','precious','Fed','inflation','黃金','白銀','聯準會','通膨']
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(r.content)
            for item in root.iter('item'):
                title = item.findtext('title', '')
                link = item.findtext('link', '')
                pub = item.findtext('pubDate', '')
                if any(k.lower() in title.lower() for k in kw):
                    items.append({'source': src, 'title': title, 'link': link, 'pub': pub[:16]})
        except Exception:
            pass
    return items[:20]

def card(label, value, hint, color="#e8eaf0"):
    bg = "#141420"
    border = "#1a1a2e"
    st.markdown(
        "<div style=\"background:" + bg + ";border:1px solid " + border + ";border-radius:10px;"
        "padding:10px;margin:4px 0;text-align:center\">"
        "<div style=\"color:#4a5a6a;font-size:10px;letter-spacing:1px\">" + label + "</div>"
        "<div style=\"color:" + color + ";font-size:18px;font-family:monospace\">" + value + "</div>"
        "<div style=\"color:#8899aa;font-size:11px\">" + hint + "</div>"
        "</div>",
        unsafe_allow_html=True
    )

def signal_info(sig, latest):
    if sig == 2:
        return 100, "🟢 強底部信號", "bull", "#00e5a0"
    elif sig == 1:
        return 60, "🟡 弱底部信號", "warn", "#ffbe00"
    elif sig == -2:
        return 100, "🔴 強頂部信號", "bear", "#ff4757"
    elif sig == -1:
        return 60, "🟡 弱頂部信號", "warn", "#ffbe00"
    else:
        return 0, "⚪ 無明確信號", "neutral", "#8899aa"

def plot_price(df, name):
    recent = df.tail(120)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent['Open'], high=recent['High'],
        low=recent['Low'], close=recent['Close'],
        increasing_line_color='#00e5a0', decreasing_line_color='#ff4757',
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['EMA_20'],
        line=dict(color='#ffa657', width=1, dash='dash'), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['EMA_50'],
        line=dict(color='#ff7b72', width=1, dash='dash'), name='EMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Upper'],
        line=dict(color='#58a6ff', width=0.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Lower'],
        line=dict(color='#58a6ff', width=0.5), fill='tonexty',
        fillcolor='rgba(88,166,255,0.05)', showlegend=False), row=1, col=1)
    for sv, mk, mc, lb in [(2,'^','#00e5a0','強底'),(1,'^','#56d364','弱底'),(-2,'v','#ff4757','強頂'),(-1,'v','#ffa198','弱頂')]:
        pts = recent[recent['Signal'] == sv]
        if len(pts):
            yv = pts['Low']*0.998 if sv > 0 else pts['High']*1.002
            fig.add_trace(go.Scatter(x=pts.index, y=yv, mode='markers',
                marker=dict(symbol='triangle-up' if sv > 0 else 'triangle-down', size=12, color=mc),
                name=lb), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['RSI_14'],
        line=dict(color='#a371f7', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_color='#ff4757', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_color='#00e5a0', line_dash='dash', opacity=0.5, row=2, col=1)
    hist_colors = ['#00e5a0' if v >= 0 else '#ff4757' for v in recent['MACD_Hist']]
    fig.add_trace(go.Bar(x=recent.index, y=recent['MACD_Hist'],
        marker_color=hist_colors, opacity=0.8, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD'],
        line=dict(color='#58a6ff', width=1), name='MACD'), row=3, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
        font=dict(color='#8899aa'),
        xaxis=dict(gridcolor='#1a1a2e'), yaxis=dict(gridcolor='#1a1a2e'),
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text=name + " | 近 120 日", font=dict(size=14, color='#c9d1d9')),
        height=520, showlegend=True,
        xaxis3=dict(rangeslider=dict(visible=False)),
    )
    return fig

def plot_ratio(gold_df, silver_df):
    ratio = (gold_df['Close'] / silver_df['Close']).dropna().tail(250)
    ma60 = ratio.rolling(60).mean()
    std60 = ratio.rolling(60).std()
    zscore = (ratio - ma60) / std60
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
        line=dict(color='#d4a843', width=2), name='金銀比'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma60.index, y=ma60,
        line=dict(color='#a8b8c8', width=1, dash='dash'), name='60日均'), row=1, col=1)
    bar_colors = ['#ff4757' if z > 0 else '#00e5a0' for z in zscore]
    fig.add_trace(go.Bar(x=zscore.index, y=zscore,
        marker_color=bar_colors, opacity=0.7, name='Z-Score'), row=2, col=1)
    fig.add_hline(y=2, line_color='#ff4757', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.add_hline(y=-2, line_color='#00e5a0', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
        font=dict(color='#8899aa'), height=400,
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text="黃金/白銀 比值分析", font=dict(size=14, color='#c9d1d9')),
    )
    return fig

def main():
    st.markdown("# 🏅 金銀轉折信號系統")
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    st.caption("資料來源: Yahoo Finance｜更新時間: " + now_str)

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
                st.warning(name + " 載入失敗: " + str(e))
        macro = fetch_macro()

    if not dfs:
        st.error("無法載入市場資料，請稍後重試。")
        return

    st.markdown("### 📊 即時報價")
    cols = st.columns(len(dfs) * 2)
    ci = 0
    for name, df in dfs.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        chg = float(latest['Close']) - float(prev['Close'])
        chg_pct = chg / float(prev['Close']) * 100
        with cols[ci]:
            st.metric(name, f"{float(latest['Close']):.2f}", delta=f"{chg:+.2f} ({chg_pct:+.2f}%)")
        ci += 1
        sig = int(latest.get('Signal', 0))
        pct, label, bias, color = signal_info(sig, latest)
        with cols[ci]:
            if pct > 0:
                st.markdown(
                    "<div style=\"padding:10px;background:#141420;border-radius:12px;"
                    "border:1px solid " + color + "44\">"
                    "<div style=\"color:" + color + ";font-size:13px;font-weight:600\">" + label + "</div>"
                    "<div style=\"color:" + color + ";font-size:22px;font-family:monospace\">" + str(pct) + "分</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style=\"padding:10px;background:#141420;border-radius:12px;"
                    "border:1px solid #1a1a2e\">"
                    "<div style=\"color:#4a5a6a;font-size:13px\">" + label + "</div>"
                    "<div style=\"color:#4a5a6a;font-size:11px\">持續監測中</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
        ci += 1

    st.divider()
    tabs = st.tabs(["📈 技術信號", "🌍 市場環境", "📰 財經新聞", "📋 COT持倉", "⚖️ 金銀比值"])

    with tabs[0]:
        for name, df in dfs.items():
            if df.empty:
                continue
            latest = df.iloc[-1]
            sig = int(latest.get('Signal', 0))
            pct, label, bias, color = signal_info(sig, latest)
            with st.expander(name + "  |  " + label, expanded=True):
                st.plotly_chart(plot_price(df, name), use_container_width=True)
                ind_cols = st.columns(4)
                rsi_val = float(latest['RSI_14'])
                k_val = float(latest['Stoch_K'])
                bb_val = float(latest['BB_Pct'])
                z_val = float(latest['ZScore_20'])
                adx_val = float(latest['ADX'])
                atr_val = float(latest['ATR_14'])
                ml_b = float(latest.get('ML_Bot', 0))
                ml_t = float(latest.get('ML_Top', 0))
                inds = [
                    ("RSI(14)", f"{rsi_val:.1f}", "超賣" if rsi_val < 30 else "超買" if rsi_val > 70 else "正常"),
                    ("KD(%K)", f"{k_val:.1f}", "超賣" if k_val < 20 else "超買" if k_val > 80 else "正常"),
                    ("布林%B", f"{bb_val:.2f}", "下軌" if bb_val < 0.1 else "上軌" if bb_val > 0.9 else "中間"),
                    ("Z-Score", f"{z_val:.2f}", "超賣區" if z_val < -2 else "超買區" if z_val > 2 else "正常"),
                    ("ADX", f"{adx_val:.1f}", "強趨勢" if adx_val > 25 else "弱趨勢"),
                    ("ATR(14)", f"{atr_val:.2f}", "波動度"),
                    ("ML底部概率", f"{ml_b:.1%}", ""),
                    ("ML頂部概率", f"{ml_t:.1%}", ""),
                ]
                for i, (lb, vl, ht) in enumerate(inds):
                    with ind_cols[i % 4]:
                        card(lb, vl, ht)
                if pct >= 60:
                    direction = "做多" if bias == 'bull' else "做空/減倉"
                    stop_side = "低點下方" if bias == 'bull' else "高點上方"
                    st.info(
                        "**操作參考** — 可留意" + direction + "機會。"
                        "止損建議設於近期" + stop_side + " 1×ATR (" + f"{atr_val:.2f}" + ")。\n\n"
                        "⚠️ 僅供參考，請搭配個人判斷與風控。"
                    )

    with tabs[1]:
        st.markdown("### 🌍 宏觀因子")
        mc = st.columns(3)
        items = [
            ('DXY','美元指數','走強→壓制黃金'),
            ('VIX','VIX恐慌','>25避險需求↑'),
            ('TIP','實質利率(TIP)','下跌→利多黃金'),
            ('TLT','長債(TLT)','上漲→利率下行'),
            ('OIL','原油(WTI)','通膨預期代理'),
            ('SPY','標普500','股市風險情緒'),
        ]
        for i, (key, lbl, hint) in enumerate(items):
            d = macro.get(key, {})
            val = d.get('value')
            chg = d.get('chg')
            with mc[i % 3]:
                if val is not None:
                    chg_color = '#00e5a0' if chg > 0 else '#ff4757'
                    st.markdown(
                        "<div style=\"background:#141420;border:1px solid #1a1a2e;"
                        "border-radius:12px;padding:14px;margin:6px 0\">"
                        "<div style=\"color:#4a5a6a;font-size:11px\">" + lbl + "</div>"
                        "<div style=\"color:#e8eaf0;font-size:20px;font-family:monospace\">" + f"{val:.2f}" + "</div>"
                        "<div style=\"color:" + chg_color + ";font-size:13px\">" + f"{chg:+.2f}%" + "</div>"
                        "<div style=\"color:#4a5a6a;font-size:11px;margin-top:4px\">" + hint + "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("### 🔍 影響解讀")
        dxy = macro.get('DXY', {})
        vix = macro.get('VIX', {})
        tip = macro.get('TIP', {})
        oil = macro.get('OIL', {})
        interps = []
        if dxy.get('chg') is not None:
            c = dxy['chg']
            if c > 0.3:
                interps.append(("🔴", "美元指數", f"上漲 {c:.2f}%，壓制黃金", "#ff4757"))
            elif c < -0.3:
                interps.append(("🟢", "美元指數", f"下跌 {abs(c):.2f}%，利多黃金", "#00e5a0"))
            else:
                interps.append(("⚪", "美元指數", f"變動 {c:.2f}%，影響中性", "#8899aa"))
        if vix.get('value') is not None:
            v = vix['value']
            if v > 25:
                interps.append(("🟡", "VIX恐慌", f"VIX={v:.1f}，避險需求上升利多黃金", "#ffbe00"))
            else:
                interps.append(("⚪", "VIX恐慌", f"VIX={v:.1f}，市場平靜", "#8899aa"))
        if tip.get('chg') is not None:
            t = tip['chg']
            if t < -0.3:
                interps.append(("🟢", "實質利率", f"TIP下跌 {abs(t):.2f}%，利多黃金", "#00e5a0"))
            elif t > 0.3:
                interps.append(("🔴", "實質利率", f"TIP上漲 {t:.2f}%，壓制黃金", "#ff4757"))
        for icon, factor, msg, color in interps:
            st.markdown(
                "<div style=\"background:#141420;border-left:3px solid " + color + ";"
                "border-radius:0 10px 10px 0;padding:10px 14px;margin:6px 0\">"
                + icon + " <b style=\"color:" + color + "\">" + factor + "</b>　" + msg +
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("### 🏦 Fed 政策對照")
        st.markdown("""
| 政策方向 | 黃金 | 白銀 |
|---------|------|------|
| 升息/鷹派 | ❌ 壓制 | ❌ 壓制 |
| 降息/鴿派 | ✅ 利多 | ✅ 利多 |
| QE 擴表 | ✅ 強烈利多 | ✅ 利多 |
| QT 縮表 | ❌ 壓制 | ❌ 壓制 |
""")
        st.caption("🔗 Fed 最新聲明: https://www.federalreserve.gov/newsevents/pressreleases.htm")

    with tabs[2]:
        st.markdown("### 📰 即時財經新聞")
        with st.spinner("載入新聞…"):
            new
