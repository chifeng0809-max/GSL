import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="金銀轉折信號", page_icon="🥇", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0a0a0f !important; color: #e8eaf0; }
[data-testid="metric-container"] { background: #141420; border: 1px solid #1a1a2e; border-radius: 12px; padding: 14px; }
div[data-testid="stMetricValue"] { font-family: monospace; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

SYMBOLS = {"GLD": "GC=F", "SLV": "SI=F"}
SYMBOL_NAMES = {"GC=F": "黃金現貨", "SI=F": "白銀現貨"}
FUTURES_MAP = {
    "GLD": {"近月": "GC=F", "次月": "GCM25.CMX", "季月": "GCU25.CMX"},
    "SLV": {"近月": "SI=F", "次月": "SIN25.CMX", "季月": "SIU25.CMX"},
}
METAL_NAMES = {"GLD": "黃金", "SLV": "白銀"}
METAL_ICONS = {"GLD": "🥇", "SLV": "🥈"}

@st.cache_data(ttl=900, show_spinner=False)
def download(symbol):
    end = datetime.today()
    start = end - timedelta(days=400)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def add_indicators(df):
    c = df['Close']
    h = df['High']
    l = df['Low']
    df['EMA_20'] = ta.trend.ema_indicator(c, 20)
    df['EMA_50'] = ta.trend.ema_indicator(c, 50)
    macd = ta.trend.MACD(c)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['RSI_14'] = ta.momentum.rsi(c, 14)
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Pct'] = bb.bollinger_pband()
    df['ATR_14'] = ta.volatility.average_true_range(h, l, c, 14)
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df['ZScore_20'] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    return df

def add_signals(df):
    bot = (df['RSI_14'] < 35) & (df['Stoch_K'] < 25) & (df['BB_Pct'] < 0.15)
    top = (df['RSI_14'] > 65) & (df['Stoch_K'] > 75) & (df['BB_Pct'] > 0.85)
    df['Signal'] = 0
    df.loc[bot, 'Signal'] = 1
    df.loc[top, 'Signal'] = -1
    df.loc[bot & (df['ZScore_20'] < -2), 'Signal'] = 2
    df.loc[top & (df['ZScore_20'] > 2), 'Signal'] = -2
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro():
    tickers = {'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'TIP': 'TIP', 'OIL': 'CL=F'}
    result = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period='5d')
            if len(hist) >= 2:
                cur = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2])
                result[name] = {'value': cur, 'chg': (cur - prev) / prev * 100}
        except Exception:
            result[name] = {'value': None, 'chg': None}
    return result

@st.cache_data(ttl=900, show_spinner=False)
def fetch_futures(key):
    symbols = FUTURES_MAP.get(key, {})
    data = {}
    for label, sym in symbols.items():
        try:
            hist = yf.download(sym, period='60d', progress=False, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if len(hist) > 0:
                data[label] = {
                    'close': hist['Close'],
                    'volume': hist['Volume'],
                    'latest': float(hist['Close'].iloc[-1]),
                    'latest_vol': float(hist['Volume'].iloc[-1]),
                }
        except Exception:
            pass
    return data

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_news():
    feeds = [
        ("Kitco", "https://www.kitco.com/rss/kitconews.rss"),
        ("MarketWatch", "https://www.marketwatch.com/rss/marketpulse"),
    ]
    items = []
    kw = ['gold', 'silver', 'metal', 'precious', 'Fed', 'inflation', 'rate', 'bullion']
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter('item'):
                title = item.findtext('title', '')
                link = item.findtext('link', '')
                pub = item.findtext('pubDate', '')[:16] if item.findtext('pubDate') else ''
                if any(k.lower() in title.lower() for k in kw):
                    items.append({'source': src, 'title': title, 'link': link, 'pub': pub})
            if len(items) >= 10:
                break
        except Exception:
            continue
    return items[:15]

def sig_label(sig):
    if sig == 2:
        return "🟢 強底部信號", "#00e5a0"
    elif sig == 1:
        return "🟡 弱底部信號", "#ffbe00"
    elif sig == -2:
        return "🔴 強頂部信號", "#ff4757"
    elif sig == -1:
        return "🟡 弱頂部信號", "#ffbe00"
    return "⚪ 無明確信號", "#8899aa"

def plot_chart(df, name):
    recent = df.tail(90)
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
        line=dict(color='#58a6ff', width=0.5),
        fill='tonexty', fillcolor='rgba(88,166,255,0.05)', showlegend=False), row=1, col=1)
    for sv, mc, lb in [(2,'#00e5a0','強底'),(1,'#56d364','弱底'),(-2,'#ff4757','強頂'),(-1,'#ffa198','弱頂')]:
        pts = recent[recent['Signal'] == sv]
        if len(pts):
            yv = pts['Low'] * 0.998 if sv > 0 else pts['High'] * 1.002
            fig.add_trace(go.Scatter(x=pts.index, y=yv, mode='markers',
                marker=dict(symbol='triangle-up' if sv > 0 else 'triangle-down', size=12, color=mc),
                name=lb), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['RSI_14'],
        line=dict(color='#a371f7', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_color='#ff4757', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_color='#00e5a0', line_dash='dash', opacity=0.5, row=2, col=1)
    hc = ['#00e5a0' if v >= 0 else '#ff4757' for v in recent['MACD_Hist']]
    fig.add_trace(go.Bar(x=recent.index, y=recent['MACD_Hist'],
        marker_color=hc, opacity=0.8, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD'],
        line=dict(color='#58a6ff', width=1), name='MACD'), row=3, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
        font=dict(color='#8899aa'),
        xaxis=dict(gridcolor='#1a1a2e'), yaxis=dict(gridcolor='#1a1a2e'),
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text=name + " | 近 90 日", font=dict(size=14, color='#c9d1d9')),
        height=500, showlegend=True,
        xaxis3=dict(rangeslider=dict(visible=False)),
    )
    return fig

def plot_futures_chart(fdata, metal_name):
    if not fdata or '近月' not in fdata:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)
    colors_map = {'近月': '#d4a843', '次月': '#58a6ff', '季月': '#a371f7'}
    for label, d in fdata.items():
        c = colors_map.get(label, '#8899aa')
        fig.add_trace(go.Scatter(
            x=d['close'].tail(60).index, y=d['close'].tail(60),
            line=dict(color=c, width=1.5), name=label), row=1, col=1)
    if '次月' in fdata and '近月' in fdata:
        s1 = fdata['近月']['close'].tail(60)
        s2 = fdata['次月']['close'].tail(60)
        spread = (s2 - s1).dropna()
        sc = ['#00e5a0' if v >= 0 else '#ff4757' for v in spread]
        fig.add_trace(go.Bar(x=spread.index, y=spread,
            marker_color=sc, opacity=0.8, name='價差(次月-近月)'), row=2, col=1)
        fig.add_hline(y=0, line_color='#8899aa', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
        font=dict(color='#8899aa'), height=420,
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text=metal_name + " 期貨結構", font=dict(size=14, color='#c9d1d9')),
        xaxis2=dict(rangeslider=dict(visible=False)),
    )
    return fig

def plot_ratio(g, s):
    ratio = (g['Close'] / s['Close']).dropna().tail(200)
    ma = ratio.rolling(60).mean()
    std = ratio.rolling(60).std()
    zs = (ratio - ma) / std
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
        line=dict(color='#d4a843', width=2), name='金銀比'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma,
        line=dict(color='#a8b8c8', width=1, dash='dash'), name='60日均'), row=1, col=1)
    bc = ['#ff4757' if z > 0 else '#00e5a0' for z in zs]
    fig.add_trace(go.Bar(x=zs.index, y=zs, marker_color=bc, opacity=0.7, name='Z-Score'), row=2, col=1)
    fig.add_hline(y=2, line_color='#ff4757', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.add_hline(y=-2, line_color='#00e5a0', line_dash='dash', opacity=0.5, row=2, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0f0f18',
        font=dict(color='#8899aa'), height=380,
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text="Gold/Silver Ratio", font=dict(size=14, color='#c9d1d9')),
    )
    return fig

def main():
    st.markdown("# 🏅 金銀轉折信號系統")
    st.caption("Yahoo Finance | 技術分析 | 每15分鐘更新")

    with st.spinner("載入資料…"):
        dfs = {}
        for key, sym in SYMBOLS.items():
            try:
                raw = download(sym)
                raw = add_indicators(raw)
                raw = add_signals(raw)
                dfs[key] = raw
            except Exception as e:
                st.warning(METAL_NAMES.get(key, key) + " 載入失敗: " + str(e))
        macro = fetch_macro()

    if not dfs:
        st.error("無法載入資料，請稍後重試。")
        return

    st.markdown("### 📊 即時報價")
    cols = st.columns(len(dfs) * 2)
    ci = 0
    for key, df in dfs.items():
        if df.empty:
            continue
        icon = METAL_ICONS.get(key, "")
        name = METAL_NAMES.get(key, key)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        chg = float(latest['Close']) - float(prev['Close'])
        chg_pct = chg / float(prev['Close']) * 100
        with cols[ci]:
            st.metric(icon + " " + name, f"{float(latest['Close']):.2f}",
                      delta=f"{chg:+.2f} ({chg_pct:+.2f}%)")
        ci += 1
        sig = int(latest.get('Signal', 0))
        label, color = sig_label(sig)
        with cols[ci]:
            st.markdown(
                "<div style=\"padding:12px;background:#141420;border-radius:12px;"
                "border:1px solid " + color + "33;text-align:center\">"
                "<div style=\"color:" + color + ";font-size:14px;font-weight:600\">" + label + "</div>"
                "</div>", unsafe_allow_html=True)
        ci += 1

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 技術信號", "🌍 市場環境", "📰 財經新聞", "⚖️ 金銀比值", "📉 期現價差"])

    with tab1:
        for key, df in dfs.items():
            if df.empty:
                continue
            icon = METAL_ICONS.get(key, "")
            name = METAL_NAMES.get(key, key)
            latest = df.iloc[-1]
            sig = int(latest.get('Signal', 0))
            label, color = sig_label(sig)
            with st.expander(icon + " " + name + "  |  " + label, expanded=True):
                st.plotly_chart(plot_chart(df, name), use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                rsi = float(latest['RSI_14'])
                k = float(latest['Stoch_K'])
                bb = float(latest['BB_Pct'])
                z = float(latest['ZScore_20'])
                atr = float(latest['ATR_14'])
                with c1:
                    st.metric("RSI(14)", f"{rsi:.1f}", delta="超賣" if rsi < 30 else ("超買" if rsi > 70 else "正常"))
                with c2:
                    st.metric("KD(%K)", f"{k:.1f}", delta="超賣" if k < 20 else ("超買" if k > 80 else "正常"))
                with c3:
                    st.metric("布林%B", f"{bb:.2f}", delta="下軌" if bb < 0.1 else ("上軌" if bb > 0.9 else "中間"))
                with c4:
                    st.metric("Z-Score", f"{z:.2f}", delta="超賣區" if z < -2 else ("超買區" if z > 2 else "正常"))
                if abs(sig) >= 1:
                    direction = "做多" if sig > 0 else "做空/減倉"
                    stop = "低點下方" if sig > 0 else "高點上方"
                    st.info("操作參考 | " + direction + " | 止損: 近期" + stop + " 1xATR=" + f"{atr:.2f}" + " | 僅供參考")

    with tab2:
        st.markdown("### 🌍 宏觀因子")
        mc = st.columns(4)
        macro_items = [
            ('DXY', '美元指數', '走強壓制黃金'),
            ('VIX', 'VIX恐慌', '>25避險需求升'),
            ('TIP', '實質利率', '下跌利多黃金'),
            ('OIL', '原油', '通膨預期代理'),
        ]
        for i, (mkey, lbl, hint) in enumerate(macro_items):
            d = macro.get(mkey, {})
            val = d.get('value')
            chg = d.get('chg')
            with mc[i]:
                if val is not None and chg is not None:
                    chg_color = '#00e5a0' if chg > 0 else '#ff4757'
                    st.markdown(
                        "<div style=\"background:#141420;border:1px solid #1a1a2e;"
                        "border-radius:12px;padding:14px;text-align:center\">"
                        "<div style=\"color:#4a5a6a;font-size:11px\">" + lbl + "</div>"
                        "<div style=\"color:#e8eaf0;font-size:20px;font-family:monospace\">" + f"{val:.2f}" + "</div>"
                        "<div style=\"color:" + chg_color + "\">" + f"{chg:+.2f}%" + "</div>"
                        "<div style=\"color:#4a5a6a;font-size:11px\">" + hint + "</div>"
                        "</div>", unsafe_allow_html=True)
        st.markdown("### 🏦 Fed 政策對照")
        st.table(pd.DataFrame({
            "政策": ["升息/鷹派", "降息/鴿派", "QE擴表", "QT縮表"],
            "黃金": ["壓制", "利多", "強烈利多", "壓制"],
            "白銀": ["壓制", "利多", "利多", "壓制"],
        }))

    with tab3:
        st.markdown("### 📰 即時財經新聞")
        with st.spinner("載入新聞…"):
            news = fetch_news()
        if news:
            for item in news:
                st.markdown(
                    "<div style=\"background:#141420;border:1px solid #1a1a2e;"
                    "border-radius:10px;padding:12px;margin:6px 0\">"
                    "<div style=\"color:#4a5a6a;font-size:10px\">" + item['source'] + "  " + item['pub'] + "</div>"
                    "<a href=\"" + item['link'] + "\" target=\"_blank\" "
                    "style=\"color:#d4a843;text-decoration:none;font-size:14px\">"
                    + item['title'] + "</a></div>", unsafe_allow_html=True)
        else:
            st.info("暫時無法載入新聞，請直接訪問：")
            st.markdown("- [Kitco Gold News](https://www.kitco.com/news/gold)")
            st.markdown("- [Reuters Commodities](https://www.reuters.com/markets/commodities/)")

    with tab4:
        st.markdown("### ⚖️ 黃金/白銀 比值")
        gdf = dfs.get('GLD')
        sdf = dfs.get('SLV')
        if gdf is not None and sdf is not None:
            rn = float(gdf['Close'].iloc[-1]) / float(sdf['Close'].iloc[-1])
            r60 = float((gdf['Close'] / sdf['Close']).tail(60).mean())
            c1, c2 = st.columns(2)
            c1.metric("當前比值", f"{rn:.1f}")
            c2.metric("60日均值", f"{r60:.1f}", delta=f"{rn-r60:+.1f}")
            st.plotly_chart(plot_ratio(gdf, sdf), use_container_width=True)
            if rn > r60 * 1.05:
                st.success("比值偏高：白銀相對低估，歷史上此時白銀常優於黃金")
            elif rn < r60 * 0.95:
                st.warning("比值偏低：黃金相對低估")
            else:
                st.info("比值正常範圍")

    with tab5:
        st.markdown("### 📉 期現貨價差 & 成交量")
        st.caption("正價差(Contango): 次月>近月 | 逆價差(Backwardation): 近月>次月")
        for key in ["GLD", "SLV"]:
            metal_name = METAL_NAMES[key]
            icon = METAL_ICONS[key]
            st.markdown("#### " + icon + " " + metal_name)
            with st.spinner(metal_name + " 期貨載入中…"):
                fdata = fetch_futures(key)
            if not fdata:
                st.warning(metal_name + " 期貨資料暫時無法取得")
                continue
            mc2 = st.columns(3)
            for i, lbl in enumerate(['近月', '次月', '季月']):
                if lbl in fdata:
                    with mc2[i]:
                        st.metric(lbl, f"{fdata[lbl]['latest']:.2f}",
                                  delta="量: " + f"{int(fdata[lbl]['latest_vol']):,}")
            if '近月' in fdata and '次月' in fdata:
                sv = fdata['次月']['latest'] - fdata['近月']['latest']
                sp = sv / fdata['近月']['latest'] * 100
                if sv > 0:
                    st.success("正價差 Contango: +" + f"{sv:.2f} (+{sp:.2f}%) | 次月>近月，正常結構")
                elif sv < 0:
                    st.error("逆價差 Backwardation: " + f"{sv:.2f} ({sp:.2f}%) | 近月>次月，現貨需求強！")
                else:
                    st.info("價差接近零，市場結構中性")
            fig2 = plot_futures_chart(fdata, metal_name)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            st.divider()

    st.caption("本系統僅供個人輔助參考，不構成任何投資建議。")

if __name__ == "__main__":
    main()
