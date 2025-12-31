import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import gspread
import os
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys
import warnings
warnings.filterwarnings('ignore')
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. 기본 설정 & CSS
# ==========================================
st.set_page_config(page_title="내 주식 비서 Pro", page_icon="📱", layout="wide")

# 5분 자동 갱신
st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        div[data-testid="stDataFrame"] { font-size: 0.8rem; }
        div.stButton > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

SHEET_NAME = "stock_db"

def configure_fonts():
    if sys.platform == 'linux':
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.isfile(font_path):
            fm.fontManager.addfont(font_path)
            plt.rc('font', family='NanumGothic')
    elif sys.platform == 'darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

configure_fonts()

# ==========================================
# 2. 유저 식별 시스템 (로그인/로그아웃)
# ==========================================
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def logout():
    st.session_state.user_id = ""
    st.cache_data.clear()
    st.rerun()

if not st.session_state.user_id:
    st.title("🚀 주식 비서 접속")
    user_input = st.text_input("사용자 이름을 입력하세요", placeholder="이름별로 데이터가 따로 저장됩니다.")
    if st.button("접속하기", use_container_width=True):
        if user_input.strip():
            st.session_state.user_id = user_input.strip()
            st.rerun()
    st.stop()

# ==========================================
# 3. 데이터 핸들링
# ==========================================
@st.cache_resource
def get_google_sheet():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(SHEET_NAME)
        return sh.sheet1
    except: return None

def load_portfolio_gs():
    sheet = get_google_sheet()
    if not sheet: return {}, {}
    try:
        data = sheet.get_all_records()
        my_p, t_i = {}, {}
        for row in data:
            if str(row.get('User')).strip() == st.session_state.user_id:
                t = str(row.get('Ticker')).strip().upper()
                if t:
                    my_p[t] = [int(row.get('Qty', 0)), float(row.get('Avg', 0))]
                    t_i[t] = [str(row.get('Name', t)), "-"]
        return my_p, t_i
    except: return {}, {}

def save_portfolio_gs(new_p, new_i):
    sheet = get_google_sheet()
    if not sheet: return
    try:
        all_d = sheet.get_all_records()
        other_d = [row for row in all_d if str(row.get('User')).strip() != st.session_state.user_id]
        final_rows = [["User", "Ticker", "Name", "Desc", "Qty", "Avg"]]
        for r in other_d:
            final_rows.append([r.get('User'), r.get('Ticker'), r.get('Name'), r.get('Desc'), r.get('Qty'), r.get('Avg')])
        for t, v in new_p.items():
            final_rows.append([st.session_state.user_id, t, new_i.get(t, [t])[0], "-", v[0], v[1]])
        sheet.update('A1', final_rows)
        st.cache_data.clear()
    except Exception as e: st.error(f"저장 실패: {e}")

my_portfolio, ticker_info = load_portfolio_gs()

@st.cache_data(ttl=50)
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            s = yf.Ticker(t)
            p = s.fast_info.get('last_price', 0.0)
            if p == 0:
                h = s.history(period="1d")
                p = h['Close'].iloc[-1] if not h.empty else 0.0
            prices[t] = p
        except: prices[t] = 0.0
    return prices

tickers = list(my_portfolio.keys())
current_prices = fetch_prices(tickers)

# ==========================================
# 4. 관리 팝업
# ==========================================
@st.dialog("📋 종목 관리")
def open_manager():
    st.caption(f"{st.session_state.user_id}님의 목록 수정")
    rows = [{"Ticker": t, "Name": ticker_info[t][0], "Qty": my_portfolio[t][0], "Avg": my_portfolio[t][1]} for t in my_portfolio]
    df_e = st.data_editor(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Ticker","Name","Qty","Avg"]), num_rows="dynamic")
    if st.button("💾 저장"):
        new_p, new_i = {}, {}
        for _, r in df_e.iterrows():
            t = str(r["Ticker"]).strip().upper()
            if t:
                new_p[t] = [int(r["Qty"]), float(r["Avg"])]
                new_i[t] = [str(r["Name"]), "-"]
        save_portfolio_gs(new_p, new_i)
        st.rerun()

# ==========================================
# 5. 메인 UI
# ==========================================
c_t, c_u = st.columns([2, 1])
c_t.subheader(f"📈 {st.session_state.user_id}님의 주식 비서")
with c_u:
    c1, c2 = st.columns(2)
    if c1.button("⚙️ 관리"): open_manager()
    if c2.button("👤 로그아웃"): logout()

menu = st.radio("메뉴", ["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"], horizontal=True, label_visibility="collapsed")
st.divider()

# [Tab 1] 자산
if menu == "📊 자산":
    # (자산 로직 생략 없이 - 이전과 동일하게 작동)
    total_ev, data = 0, []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        ev = c * q; bv = a * q; pct = ((ev - bv) / bv * 100) if bv > 0 else 0
        total_ev += ev
        data.append({"종목": f"{ticker_info[t][0]}({t})", "현재가": c, "수익률": pct, "평가액": ev})
    st.metric("총 평가액", f"${total_ev:,.2f}")
    if data: st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

# [Tab 2] AI 예측
elif menu == "🔮 AI예측":
    if tickers:
        sel = st.selectbox("종목 선택", tickers)
        if st.button("🤖 예측 실행"):
            df_h = yf.download(sel, period="1y", progress=False)
            y = df_h['Close'].values.ravel()
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            pred = model.predict(np.arange(len(y), len(y)+30).reshape(-1, 1))
            st.metric("30일 뒤 예상", f"${pred[-1]:.2f}")
            st.line_chart(np.append(y, pred))

# [Tab 3] 종합분석 (생략되었던 부분 복구)
elif menu == "📉 종합분석":
    if tickers:
        sel = st.selectbox("진단할 종목", tickers)
        if st.button("🔍 상세 진단"):
            with st.spinner("재무제표 분석 중..."):
                info = yf.Ticker(sel).info
                c1, c2, c3 = st.columns(3)
                c1.metric("PER", f"{info.get('trailingPE', 0):.2f}")
                c2.metric("PBR", f"{info.get('priceToBook', 0):.2f}")
                c3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
                st.write(f"**기업 개요:** {info.get('longBusinessSummary', '정보 없음')[:300]}...")

# [Tab 4] 스캔 (생략되었던 부분 복구)
elif menu == "📡 스캔":
    if st.button("🚀 전체 종목 스캔"):
        with st.spinner("RSI 및 급등주 찾는 중..."):
            df = yf.download(tickers, period="2mo", progress=False)['Close']
            res = []
            for t in tickers:
                c = df[t].dropna(); p = c.iloc[-1]
                pct = (p - c.iloc[-2])/c.iloc[-2]*100
                diff = c.diff(); up = diff.clip(lower=0).rolling(14).mean(); down = -diff.clip(upper=0).rolling(14).mean()
                rsi = 100 - (100/(1 + up/down)).iloc[-1]
                sig = "🔥급등" if pct >= 3 else ("💎과매도" if rsi <= 30 else "")
                if sig: res.append([t, f"{pct:+.2f}%", f"{rsi:.1f}", sig])
            st.dataframe(pd.DataFrame(res, columns=["티커", "등락", "RSI", "신호"]) if res else "특이사항 없음")

# [Tab 5] 뉴스 (강화된 감성 분석 버전)
elif menu == "📰 뉴스":
    # (앞서 설명한 강화된 뉴스 분석 로직 전체 포함)
    pos_dict = {'상승':1, '호재':2, '급등':3, '수익':1, '최고':2, '흑자':2}
    neg_dict = {'하락':-1, '악재':-2, '급락':-3, '손실':-1, '적자':-2}
    if st.button("🌍 AI 뉴스 분석"):
        items = []
        for t in tickers:
            q = urllib.parse.quote(f"{ticker_info[t][0]} {t}")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko")
            if feed.entries:
                e = feed.entries[0]; score = 0
                for w, v in pos_dict.items(): score += v if w in e.title else 0
                for w, v in neg_dict.items(): score += v if w in e.title else 0
                items.append({"종목": ticker_info[t][0], "분석": "😊" if score>0 else ("😨" if score<0 else "🤔"), "제목": e.title, "링크": e.link})
        st.dataframe(pd.DataFrame(items), column_config={"링크": st.column_config.LinkColumn("🔗")}, hide_index=True)
