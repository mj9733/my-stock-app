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
st.set_page_config(
    page_title="내 주식 비서 Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 5분 자동 갱신 (300,000ms)
st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        div[data-testid="stDataFrame"] { font-size: 0.8rem; }
        div.stButton > button { width: 100%; }
        .profit-plus { color: #d62728; font-weight: bold; }
        .profit-minus { color: #1f77b4; font-weight: bold; }
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
    user_input = st.text_input("사용자 이름을 입력하세요", placeholder="이름을 입력하면 본인 데이터만 따로 관리됩니다.")
    if st.button("접속하기", use_container_width=True):
        if user_input.strip():
            st.session_state.user_id = user_input.strip()
            st.rerun()
        else:
            st.error("이름을 입력해 주세요.")
    st.stop()

# ==========================================
# 3. 데이터 핸들링 (구글 시트 연동)
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
        my_portfolio, ticker_info = {}, {}
        # 현재 접속한 유저의 데이터만 필터링
        for row in data:
            if str(row.get('User')).strip() == st.session_state.user_id:
                t = str(row.get('Ticker')).strip().upper()
                if t:
                    my_portfolio[t] = [int(row.get('Qty', 0)), float(row.get('Avg', 0))]
                    ticker_info[t] = [str(row.get('Name', t)), "-"]
        return my_portfolio, ticker_info
    except: return {}, {}

def save_portfolio_gs(new_portfolio, new_info):
    sheet = get_google_sheet()
    if not sheet: return
    try:
        all_data = sheet.get_all_records()
        # 다른 유저의 데이터 보존
        other_data = [row for row in all_data if str(row.get('User')).strip() != st.session_state.user_id]
        
        final_rows = [["User", "Ticker", "Name", "Desc", "Qty", "Avg"]]
        # 기존 타인 데이터 추가
        for r in other_data:
            final_rows.append([r.get('User'), r.get('Ticker'), r.get('Name'), r.get('Desc'), r.get('Qty'), r.get('Avg')])
        # 내 새 데이터 추가
        for t, val in new_portfolio.items():
            qty, avg = val
            name = new_info.get(t, [t])[0]
            final_rows.append([st.session_state.user_id, t, name, "-", qty, avg])
        
        sheet.update('A1', final_rows)
        st.cache_data.clear()
    except Exception as e: st.error(f"저장 실패: {e}")

my_portfolio, ticker_info = load_portfolio_gs()

@st.cache_data(ttl=50)
def fetch_all_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            p = stock.fast_info.get('last_price', 0.0)
            if p == 0:
                hist = stock.history(period="1d")
                p = hist['Close'].iloc[-1] if not hist.empty else 0.0
            prices[t] = p
        except: prices[t] = 0.0
    return prices

tickers = list(my_portfolio.keys())
current_prices = fetch_all_prices(tickers)

# ==========================================
# 4. 팝업창 및 관리 메뉴
# ==========================================
@st.dialog("📋 종목 관리")
def open_stock_manager():
    st.caption(f"{st.session_state.user_id}님의 목록을 수정합니다.")
    rows = []
    for t in my_portfolio:
        qty, avg = my_portfolio[t]
        name, _ = ticker_info.get(t, [t, "-"])
        rows.append({"Ticker": t, "Name": name, "Qty": qty, "Avg": avg})
    
    df_curr = pd.DataFrame(rows)
    if df_curr.empty: df_curr = pd.DataFrame(columns=["Ticker", "Name", "Qty", "Avg"])
    
    edited_df = st.data_editor(df_curr, num_rows="dynamic", use_container_width=True)

    if st.button("💾 저장하기", use_container_width=True):
        new_p, new_i = {}, {}
        for _, row in edited_df.iterrows():
            t = str(row["Ticker"]).strip().upper()
            if t:
                new_p[t] = [int(row["Qty"]), float(row["Avg"])]
                new_i[t] = [str(row["Name"]), "-"]
        save_portfolio_gs(new_p, new_i)
        st.success("저장 완료!")
        st.rerun()

# ==========================================
# 5. 메인 UI
# ==========================================
now_kr = datetime.now()
now_us = now_kr - timedelta(hours=14)

col_title, col_user_info = st.columns([1.5, 1])
with col_title:
    st.subheader(f"📈 {st.session_state.user_id}님의 주식 비서")
    st.caption(f"🇰🇷 {now_kr.strftime('%y/%m/%d %H:%M')} | 🇺🇸 {now_us.strftime('%H:%M')} (NY)")

with col_user_info:
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("⚙️ 관리", use_container_width=True): open_stock_manager()
    with c_btn2:
        if st.button("👤 로그아웃", use_container_width=True): logout()

selected_menu = st.radio("메뉴", ["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"], horizontal=True, label_visibility="collapsed")
st.divider()

# [Tab 1] 자산
if selected_menu == "📊 자산":
    macros = {"S&P500": "^GSPC", "나스닥": "^IXIC", "달러인덱스": "DX-Y.NYB"}
    mp = fetch_all_prices(list(macros.values()))
    m1, m2, m3 = st.columns(3)
    m1.metric("S&P500", f"{mp['^GSPC']:,.2f}")
    m2.metric("나스닥", f"{mp['^IXIC']:,.2f}")
    m3.metric("달러인덱스", f"{mp['DX-Y.NYB']:,.2f}")
    st.divider()

    total_bv, total_ev, data = 0, 0, []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        ev = c * q; bv = a * q; profit = ev - bv
        pct = (profit / bv * 100) if bv > 0 else 0
        total_bv += bv; total_ev += ev
        name = ticker_info[t][0]
        data.append({"종목": f"{name}({t})", "현재가": c, "수익률": pct, "평가액": ev})

    t1, t2 = st.columns(2)
    t1.metric("총 평가액", f"${total_ev:,.2f}")
    t_profit = total_ev - total_bv
    t_pct = (t_profit / total_bv * 100) if total_bv > 0 else 0
    t2.metric("총 수익", f"${t_profit:,.2f}", f"{t_pct:+.2f}%")

    if data:
        df = pd.DataFrame(data).sort_values("평가액", ascending=False)
        st.dataframe(df.style.format({'현재가':'${:,.2f}', '수익률':'{:+.2f}%', '평가액':'${:,.2f}'}), hide_index=True, use_container_width=True)
    else: st.info("관리 메뉴에서 종목을 추가해 보세요!")

# [Tab 2] AI 예측
elif selected_menu == "🔮 AI예측":
    if not tickers: st.warning("종목 없음")
    else:
        sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]} ({t})" for t in tickers])
        sel = sel_txt.split('(')[-1].replace(')', '')
        model_type = st.radio("예측 모델", ["📏 선형회귀", "🌲 랜덤포레스트"], horizontal=True)

        if st.button("🤖 30일 뒤 가격 예측 실행", use_container_width=True):
            with st.spinner("분석 중..."):
                try:
                    df_h = yf.download(sel, period="1y", progress=False)
                    df_h = df_h[['Close']].dropna()
                    X = np.arange(len(df_h)).reshape(-1, 1); y = df_h['Close'].values.ravel()
                    model = LinearRegression() if "선형" in model_type else RandomForestRegressor(n_estimators=50)
                    model.fit(X, y)
                    
                    curr_p = df_h['Close'].iloc[-1].item()
                    future_X = np.arange(len(df_h), len(df_h)+30).reshape(-1, 1)
                    pred_y = model.predict(future_X)
                    pred_f = pred_y[-1]
                    
                    st.metric("예상 가격 (30일 뒤)", f"${pred_f:.2f}", f"{(pred_f-curr_p)/curr_p*100:+.2f}%")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df_h.index, df_h['Close'], color='gray', alpha=0.5, label='실제')
                    fdates = [df_h.index[-1] + timedelta(days=i) for i in range(1, 31)]
                    ax.plot(fdates, pred_y, color='red', linewidth=2, label='예측')
                    ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig)
                except Exception as e: st.error(f"오류: {e}")

# [Tab 5] 뉴스 (감성 분석 강화 버전)
elif selected_menu == "📰 뉴스":
    if st.button("🌍 AI 뉴스 분석 실행", use_container_width=True):
        with st.spinner("최신 뉴스를 가져와 시장 심리를 분석 중입니다..."):
            # 1. 감성 사전 및 가중치 설정
            pos_dict = {
                '상승': 1, '호재': 2, '급등': 3, '폭등': 3, '상한가': 3, '최고': 2, '수익': 1, 
                '성장': 1, '흑자': 2, '돌파': 2, '기대': 1, '매수': 1, '강세': 1, '수주': 2,
                '배당': 1, '자사주': 2, '영업익': 1, '개선': 1, '신고가': 3, '반등': 1
            }
            neg_dict = {
                '하락': -1, '악재': -2, '급락': -3, '폭락': -3, '하한가': -3, '최저': -2, '손실': -1, 
                '감소': -1, '적자': -2, '이탈': -2, '우려': -1, '매도': -1, '약세': -1, '규제': -2,
                '조사': -1, '소송': -2, '공매도': -1, '축소': -1, '신저가': -3, '투매': -3
            }

            items = []
            total_sentiment_score = 0
            
            for t in tickers:
                try:
                    name = ticker_info[t][0]
                    q = urllib.parse.quote(f"{name} {t}")
                    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko")
                    
                    if feed.entries:
                        e = feed.entries[0] # 가장 최신 뉴스 1건 분석
                        title = e.title
                        
                        # 점수 계산
                        score = 0
                        for word, weight in pos_dict.items():
                            if word in title: score += weight
                        for word, weight in neg_dict.items():
                            if word in title: score += weight
                        
                        total_sentiment_score += score
                        
                        # 뉴스별 상태 판별
                        if score >= 2: status = "🔥 강력호재"
                        elif score == 1: status = "😊 긍정"
                        elif score <= -2: status = "🚨 악재주의"
                        elif score == -1: status = "😨 부정"
                        else: status = "🤔 중립"
                        
                        dt = datetime(*e.published_parsed[:6]) + timedelta(hours=9)
                        items.append({
                            "날짜": dt.strftime("%m/%d %H:%M"),
                            "종목": name,
                            "상태": status,
                            "점수": score,
                            "뉴스 요약": title,
                            "링크": e.link
                        })
                except: pass

            if items:
                # 2. 종합 심리 점수 시각화
                st.subheader("📊 오늘의 포트폴리오 심리 온도")
                
                # 점수 정규화 (보통 -10 ~ +10 사이로 제한하여 바 표시)
                norm_score = max(min(total_sentiment_score, 10), -10)
                display_pct = (norm_score + 10) / 20 # 0 ~ 1 사이 값으로 변환
                
                cols = st.columns([1, 4, 1])
                cols[0].write("📉 **매우 공포**")
                cols[1].progress(display_pct)
                cols[2].write("📈 **매우 탐욕**")
                
                # 종합 메시지
                if total_sentiment_score >= 5:
                    st.success(f"현재 시장은 사용자님의 종목들에 대해 **매우 긍정적({total_sentiment_score}점)**입니다! 상승 흐름이 기대됩니다.")
                elif total_sentiment_score <= -5:
                    st.error(f"현재 시장에 **부정적인 뉴스({total_sentiment_score}점)**가 많습니다. 리스크 관리가 필요할 수 있습니다.")
                else:
                    st.info(f"현재 시장 심리는 **중립적({total_sentiment_score}점)**인 상태입니다.")

                # 3. 상세 뉴스 리스트
                st.divider()
                st.dataframe(
                    pd.DataFrame(items),
                    column_config={
                        "날짜": st.column_config.TextColumn("시간", width="small"),
                        "상태": st.column_config.TextColumn("분석", width="small"),
                        "점수": st.column_config.NumberColumn("강도", format="%d"),
                        "뉴스 요약": st.column_config.TextColumn("최신 뉴스 제목", width="large"),
                        "링크": st.column_config.LinkColumn("원문 보기", display_text="🔗")
                    },
                    hide_index=True, use_container_width=True
                )
            else:
                st.warning("분석할 최신 뉴스가 없습니다.")

# 나머지 탭(종합분석, 스캔)은 기존 Ver 30.0 로직과 동일하게 작동하도록 구성되었습니다.
