import streamlit as st
import requests
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

# [추가] 브라우저처럼 위장하여 차단을 피하는 세션 함수
def get_safe_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    })
    return session

# [추가] 재무 정보 호출 시 1시간 동안 결과를 기억하여 서버 부하 감소
@st.cache_data(ttl=3600)
def fetch_safe_financials(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol, session=get_safe_session())
        return t.info
    except:
        return {}
# ==========================================
# 1. 기본 설정 & CSS
# ==========================================
st.set_page_config(page_title="내 주식 비서 Pro", page_icon="📱", layout="wide")

# 5분 자동 갱신
st_autorefresh(interval=60 * 60 * 1000, key="data_refresh")

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
# 3. 팝업창 (매뉴얼 및 관리)
# ==========================================
@st.dialog("📖 주식 비서 사용 매뉴얼")
def show_manual():
    st.write("### 🚀 주요 기능 설명")
    st.markdown("""
    1. **📊 자산:** 내 포트폴리오의 실시간 평가액과 수익률을 확인합니다. 수익률 순 정렬이 가능합니다.
    2. **🔮 AI예측:** 과거 1년치 데이터를 학습하여 향후 30일간의 주가 흐름을 선형/비선형 모델로 예측합니다.
    3. **📉 종합분석:** 기업의 재무 건전성을 분석하고 투자 적정성을 평가합니다.
    4. **📡 스캔:** 전 종목의 등락률과 RSI 지표를 계산하여 매수/매도 타이밍을 포착합니다.
    5. **📰 뉴스:** AI가 뉴스 제목의 키워드를 분석하여 시장 심리(긍정/부정)를 점수로 환산합니다.
    """)
    
    st.divider()
    
    st.write("### 💡 주식 용어 사전")
    st.markdown("""
    * **PER (주가수익비율):** 시가총액을 순이익으로 나눈 값입니다. 보통 20보다 낮으면 저평가로 봅니다.
    * **PBR (주가순자산비율):** 주가가 기업의 자산에 비해 얼마나 비싼지 나타냅니다. 1.5 미만이면 자산 가치가 우수합니다.
    * **ROE (자기자본이익률):** 내 돈으로 얼마나 돈을 잘 벌었는지 나타내는 수익성 지표입니다. 15% 이상이면 우량합니다.
    * **RSI (상대강도지수):** 과매수/과매도를 판단하는 지표입니다.
    """)
    # LaTeX를 사용한 RSI 공식 설명
    st.latex(r"RSI = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}")
    st.caption("※ RSI가 30 이하이면 '과매도(매수 기회)', 70 이상이면 '과매수(주의)'로 해석합니다.")

# (load_portfolio_gs, save_portfolio_gs 등 데이터 핸들링 로직은 이전과 동일)

# ==========================================
# 5. 메인 UI (상단 버튼 배치 수정)
# ==========================================
now_kr = datetime.now()
now_us = now_kr - timedelta(hours=14) # 서머타임 미적용 기준 14시간 차이

col_title, col_user_btns = st.columns([1.5, 1])
with col_title:
    st.subheader(f"📈 {st.session_state.user_id}님의 주식 비서")
    # 한국 및 미국 시간 표시 복구
    st.caption(f"🇰🇷 {now_kr.strftime('%y/%m/%d %H:%M')} | 🇺🇸 {now_us.strftime('%H:%M')} (NY)")

with col_user_btns:
    # 3개의 버튼을 가로로 나란히 배치
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("📖 매뉴얼", use_container_width=True): show_manual()
    with btn_col2:
        if st.button("⚙️ 관리", use_container_width=True): open_stock_manager() # 이전 다이얼로그 함수
    with btn_col3:
        if st.button("👤 로그아웃", use_container_width=True): logout()
            
menu = st.radio("메뉴", ["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"], horizontal=True, label_visibility="collapsed")
st.divider()

# [Tab 1] 자산 (수익률 표시 복구 버전)
if menu == "📊 자산":
    total_ev, total_bv, data = 0, 0, []
    
    for t in tickers:
        q, a = my_portfolio[t]
        c = current_prices.get(t, 0)
        
        ev = c * q  # 현재 평가액
        bv = a * q  # 총 매수 금액
        profit = ev - bv
        pct = (profit / bv * 100) if bv > 0 else 0
        
        total_ev += ev
        total_bv += bv
        
        data.append({
            "종목": f"{ticker_info[t][0]}({t})",
            "현재가": c,
            "수익률": pct,
            "평가액": ev
        })

    # 총 수익금 및 수익률 계산
    total_profit = total_ev - total_bv
    total_pct = (total_profit / total_bv * 100) if total_bv > 0 else 0
    
    # 델타(수정치)를 포함한 메트릭 표시
    st.metric(
        label="총 자산 평가액", 
        value=f"${total_ev:,.2f}", 
        delta=f"${total_profit:,.2f} ({total_pct:+.2f}%)"
    )
    
    if data:
        df = pd.DataFrame(data).sort_values("평가액", ascending=False)
        st.dataframe(
            df.style.format({
                '현재가': '${:,.2f}', 
                '수익률': '{:+.2f}%', 
                '평가액': '${:,.2f}'
            }), 
            hide_index=True, 
            use_container_width=True
        )

# [Tab 2] AI 예측 (GBR & SVR 추가 및 Alpha Vantage 연동 버전)
elif menu == "🔮 AI예측":
    st.warning("⚠️ **AI 예측은 과거 데이터를 기반으로 한 기술적 분석이며, 실제 투자 결과는 시장 상황에 따라 다를 수 있습니다. 재미와 참고용으로만 활용해 주세요.**")
    
    if not tickers:
        st.info("종목이 없습니다. 관리 메뉴에서 종목을 먼저 추가해 주세요.")
    else:
        c_sel, c_opt = st.columns([1.5, 1.5])
        with c_sel:
            sel_txt = st.selectbox("예측할 종목 선택", [f"{ticker_info[t][0]} ({t})" for t in tickers], label_visibility="collapsed")
            sel = sel_txt.split('(')[-1].replace(')', '')
        with c_opt:
            # 더 정교한 분석을 위한 모델 라인업 확장
            model_type = st.selectbox("분석 모델 선택", 
                ["📏 선형회귀", "🌲 랜덤포레스트", "📈 Gradient Boosting (추천)", "🎯 SVR (비선형 분석)"], 
                label_visibility="collapsed")

        if st.button("🤖 고성능 AI 미래 가격 예측 실행", use_container_width=True):
            with st.spinner(f"{model_type} 모델 학습 및 분석 중..."):
                try:
                    # 1. Alpha Vantage를 통한 데이터 수집 (안전 버전)
                    # 이전 단계에서 만든 fetch_history_av 함수를 사용한다고 가정합니다.
                    df_h = fetch_history_av(sel) 
                    
                    if df_h.empty:
                        # Alpha Vantage 실패 시 야후 세션 방식으로 백업
                        df_h = yf.download(sel, period="1y", session=get_safe_session(), progress=False)
                        df_h = df_h[['Close']].dropna()

                    if df_h.empty: raise Exception("데이터를 불러올 수 없습니다.")

                    # 2. 데이터 전처리
                    X = np.arange(len(df_h)).reshape(-1, 1)
                    y = df_h['Close'].values.ravel()
                    
                    # SVR과 Gradient Boosting을 위한 스케일링 준비
                    from sklearn.preprocessing import StandardScaler
                    scaler_X = StandardScaler().fit(X)
                    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
                    
                    X_scaled = scaler_X.transform(X)
                    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

                    # 3. 모델 선택 및 학습
                    if "선형" in model_type:
                        model = LinearRegression()
                        model.fit(X, y)
                    elif "랜덤" in model_type:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                    elif "Gradient" in model_type:
                        # 오차를 순차적으로 보정하여 추세 파악에 탁월함
                        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                        model.fit(X, y)
                    elif "SVR" in model_type:
                        # 비선형적인 주가 흐름을 파악하는 데 강력함
                        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
                        model.fit(X_scaled, y_scaled)

                    # 4. 미래 30일 예측
                    future_days = 30
                    future_X = np.arange(len(df_h), len(df_h) + future_days).reshape(-1, 1)
                    
                    if "SVR" in model_type:
                        future_X_scaled = scaler_X.transform(future_X)
                        pred_y_scaled = model.predict(future_X_scaled)
                        pred_y = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).ravel()
                        trend_line_scaled = model.predict(X_scaled)
                        trend_line = scaler_y.inverse_transform(trend_line_scaled.reshape(-1, 1)).ravel()
                    else:
                        pred_y = model.predict(future_X)
                        trend_line = model.predict(X)

                    # 5. 결과 시각화
                    curr_p = y[-1]
                    pred_f = pred_y[-1]
                    pct = (pred_f - curr_p) / curr_p * 100
                    
                    st.metric(f"30일 뒤 예상 ({model_type})", f"${pred_f:.2f}", f"{pct:+.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df_h.index, y, label='실제 주가', color='gray', alpha=0.5)
                    ax.plot(df_h.index, trend_line, '--', label='AI 분석 추세', color='orange', alpha=0.7)
                    
                    last_dt = df_h.index[-1]
                    fdates = [last_dt + timedelta(days=i) for i in range(1, future_days + 1)]
                    ax.plot(fdates, pred_y, 'r-', linewidth=2, label='미래 예측')
                    
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y.%m"))
                    ax.legend()
                    ax.grid(True, alpha=0.3, linestyle='--')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"예측 도중 오류가 발생했습니다: {e}")

# [Tab 3] 종합분석 (개정본)
elif menu == "📉 종합분석":
    if not tickers:
        st.warning("분석할 종목이 없습니다.")
    else:
        sel_txt = st.selectbox("진단할 종목", [f"{ticker_info[t][0]} ({t})" for t in tickers])
        sel_ticker = sel_txt.split('(')[-1].replace(')', '')
        
        if st.button("🔍 상세 재무 진단 실행", use_container_width=True):
            with st.spinner("야후 서버에서 재무 데이터를 가져오는 중..."):
                info = fetch_safe_financials(sel_ticker)
                
                if not info:
                    st.error("현재 야후 서버 접속이 일시적으로 제한되었습니다. 잠시 후 다시 시도해 주세요.")
                else:
                    per = info.get('trailingPE', 0)
                    pbr = info.get('priceToBook', 0)
                    roe = info.get('returnOnEquity', 0)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("PER", f"{per:.2f}" if per else "정보 없음")
                    c2.metric("PBR", f"{pbr:.2f}" if pbr else "정보 없음")
                    c3.metric("ROE", f"{roe*100:.2f}%" if roe else "정보 없음")
                    st.write(f"**기업 요약:** {info.get('longBusinessSummary', '설명이 없습니다.')[:500]}...")

                    # 1. 주요 지표 표시 (Metric)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("PER (주가수익비율)", f"{per:.2f}" if per else "N/A")
                    c2.metric("PBR (주가순자산비율)", f"{pbr:.2f}" if pbr else "N/A")
                    c3.metric("ROE (자기자본이익률)", f"{roe*100:.2f}%" if roe else "N/A")
                    
                    # 2. 투자 의견 자동 생성
                    st.divider()
                    score = 0
                    if per and 0 < per < 20: score += 1
                    if pbr and 0 < pbr < 1.5: score += 1
                    if roe and roe > 0.15: score += 1
                    
                    status = "🟢 양호" if score >= 2 else ("🟡 보통" if score == 1 else "🔴 관망")
                    st.subheader(f"AI 종합 진단 결과: {status}")
                    
                    # 3. 기업 개요 (접이식으로 깔끔하게)
                    with st.expander("🏢 기업 개요 보기"):
                        st.write(biz_summary)

                    # 4. 분기 실적 차트
                    try:
                        ticker_obj = yf.Ticker(sel_ticker, session=get_safe_session())
                        fin = ticker_obj.quarterly_financials
                        if not fin.empty:
                            st.write("### 📊 최근 분기 실적 추이")
                            st.bar_chart(fin.loc['Total Revenue'])
                    except:
                        st.caption("실적 차트를 불러올 수 없습니다.")

# [Tab 4] 스캔 (개정본)
elif menu == "📡 스캔":
    if st.button("🚀 전체 종목 기술적 지표 스캔", use_container_width=True):
        if not tickers:
            st.warning("종목이 없습니다.")
        else:
            with st.spinner("RSI 및 변동률 분석 중..."):
                try:
                    # 세션을 사용하여 차단 방지
                    df_all = yf.download(tickers, period="2mo", session=get_safe_session(), progress=False)
                    res = []
                    
                    for t in tickers:
                        # 종목별 데이터 추출
                        ticker_data = df_all[t] if len(tickers) > 1 else df_all
                        c = ticker_data['Close'].dropna()
                        
                        # [핵심] 데이터가 부족하거나 없으면 건너뛰어 에러 방지
                        if c.empty or len(c) < 15: 
                            continue
                        
                        # 지표 계산
                        p_now = c.iloc[-1]
                        p_prev = c.iloc[-2]
                        pct = (p_now - p_prev) / p_prev * 100
                        
                        # RSI 계산
                        diff = c.diff()
                        up = diff.clip(lower=0).rolling(14).mean()
                        down = -diff.clip(upper=0).rolling(14).mean()
                        rsi = 100 - (100 / (1 + (up / down).iloc[-1]))
                        
                        sig = "🔥급등" if pct >= 3 else ("💎과매도" if rsi <= 30 else "")
                        res.append([t, f"{pct:+.2f}%", f"{rsi:.1f}", sig])
                    
                    if res:
                        st.dataframe(pd.DataFrame(res, columns=["티커", "등락", "RSI", "신호"]), use_container_width=True)
                    else:
                        st.info("현재 분석 가능한 데이터가 부족합니다.")
                except Exception as e:
                    st.error("데이터를 불러오는 중 문제가 발생했습니다. 관리자 설정을 확인하세요.")

# [Tab 5] 뉴스 분석 (에러 방지 및 감성 분석 강화 버전)
elif menu == "📰 뉴스":
    st.info("🌍 AI가 실시간 뉴스를 분석하여 시장의 긍정/부정 심리를 점수화합니다. (20분 단위 갱신)")
    
    if not tickers:
        st.warning("분석할 종목이 없습니다. 관리 메뉴에서 종목을 추가해 주세요.")
    else:
        if st.button("🌍 최신 뉴스 감성 분석 실행", use_container_width=True):
            with st.spinner("보유 종목 관련 최신 뉴스를 수집하고 분석 중입니다..."):
                try:
                    # 1. 감성 사전 및 가중치 설정
                    pos_dict = {'상승':1, '호재':2, '급등':3, '폭등':3, '수익':1, '최고':2, '흑자':2, '돌파':1, '배당':1, '성장':1}
                    neg_dict = {'하락':-1, '악재':-2, '급락':-3, '폭락':-3, '손실':-1, '적자':-2, '우려':-1, '이탈':-1, '규제':-2, '적자':-2}

                    items = []
                    total_sentiment_score = 0
                    
                    for t in tickers:
                        try:
                            # 종목명과 티커로 검색 쿼리 생성
                            stock_name = ticker_info[t][0]
                            q = urllib.parse.quote(f"{stock_name} {t}")
                            
                            # Google News RSS 피드 가져오기
                            feed_url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
                            feed = feedparser.parse(feed_url)
                            
                            # [핵심] 뉴스 검색 결과가 있는지 확인하여 IndexError 방지
                            if not feed.entries:
                                continue
                                
                            # 가장 최신 뉴스 1건 분석
                            e = feed.entries[0]
                            title = e.title
                            
                            # 감성 점수 계산
                            score = 0
                            for word, weight in pos_dict.items():
                                if word in title: score += weight
                            for word, weight in neg_dict.items():
                                if word in title: score += weight
                            
                            total_sentiment_score += score
                            
                            # 상태 판별
                            if score >= 2: status = "🔥 강력호재"
                            elif score == 1: status = "😊 긍정"
                            elif score <= -2: status = "🚨 악재주의"
                            elif score == -1: status = "😨 부정"
                            else: status = "🤔 중립"
                            
                            # 날짜 처리 (KST 기준)
                            dt = datetime(*e.published_parsed[:6]) + timedelta(hours=9)
                            
                            items.append({
                                "시간": dt.strftime("%m/%d %H:%M"),
                                "종목": stock_name,
                                "심리": status,
                                "점수": score,
                                "뉴스 제목": title,
                                "링크": e.link
                            })
                        except Exception:
                            # 개별 뉴스 처리 실패 시 해당 종목만 건너뜀
                            continue

                    if items:
                        # 2. 종합 심리 지수 표시
                        st.subheader("📊 오늘의 포트폴리오 심리 온도")
                        
                        # 점수를 0~1 사이로 정규화하여 바(Bar) 표시
                        norm_score = max(min(total_sentiment_score, 10), -10)
                        gauge_val = (norm_score + 10) / 20 
                        
                        c1, c2, c3 = st.columns([1, 4, 1])
                        c1.write("📉 **매우 공포**")
                        c2.progress(gauge_val)
                        c3.write("📈 **매우 탐욕**")
                        
                        # 3. 상세 결과 표
                        st.divider()
                        df_news = pd.DataFrame(items)
                        st.dataframe(
                            df_news,
                            column_config={
                                "점수": st.column_config.NumberColumn("강도", format="%d"),
                                "링크": st.column_config.LinkColumn("원문", display_text="🔗")
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.warning("현재 보유 종목에 대한 최신 뉴스를 찾을 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"뉴스 수집 중 오류가 발생했습니다: {e}")
