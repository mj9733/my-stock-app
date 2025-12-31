import streamlit as st
import pandas as pd
import numpy as np
import requests
import gspread
import yfinance as yf  # 누락되었던 임포트 추가
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import feedparser
import urllib.parse
import time
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from streamlit_autorefresh import st_autorefresh
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 기본 설정 및 보안 세션 함수
# ==========================================
st.set_page_config(page_title="주식 비서 Polygon Pro", page_icon="🛡️", layout="wide")
st_autorefresh(interval=60 * 60 * 1000, key="data_refresh") # 1시간 갱신

POLYGON_KEY = st.secrets["polygon_key"]
SHEET_NAME = "stock_db"

def get_safe_session():
    """야후 차단을 피하기 위한 브라우저 위장 세션"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    })
    return session

@st.cache_data(ttl=3600)
def fetch_safe_financials(symbol):
    """안전한 방식으로 재무 정보 가져오기"""
    try:
        t = yf.Ticker(symbol, session=get_safe_session())
        return t.info
    except: return {}

# ==========================================
# 2. Polygon 데이터 엔진
# ==========================================
@st.cache_data(ttl=3600)
def fetch_history_polygon(symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df[['o', 'h', 'l', 'c', 'v']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_current_price_polygon(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        if "results" in data: return float(data["results"][0]["c"])
        return 0.0
    except: return 0.0

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
    st.title("🔐 주식 비서 접속")
    u_input = st.text_input("이름을 입력하세요", placeholder="이름별로 데이터가 따로 저장됩니다.")
    if st.button("접속"):
        st.session_state.user_id = u_input.strip()
        st.rerun()
    st.stop()

my_portfolio, ticker_info = load_portfolio_gs()
tickers = list(my_portfolio.keys())

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
# 5. 팝업창 (매뉴얼 및 관리)
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

    st.write("### 🤖 고성능 AI 모델 설명")
    st.markdown("""
    * **Gradient Boosting:** 여러 트리를 결합해 오차를 줄이는 최신 모델 (추천)
    * **SVR:** 주가의 비선형적 파동을 분석하는 데 탁월함
    * **성공률:** 과거 30일 전 데이터로 현재가를 얼마나 맞췄는지 나타내는 지표
    """)
    st.info("Polygon 무료 API 정책에 따라 주가는 전일 종가 기준으로 표시됩니다.")

# ==========================================
# 6. 메인 UI 및 듀얼 시계
# ==========================================
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if not st.session_state.user_id:
    st.title("🔐 주식 비서 Polygon 접속")
    u_input = st.text_input("이름을 입력하세요")
    if st.button("접속"):
        st.session_state.user_id = u_input.strip()
        st.rerun()
    st.stop()

c_t, c_b = st.columns([1.5, 1.2])
with c_t:
    st.subheader(f"📈 {st.session_state.user_id}님의 인텔리전트 비서")
    now_kr = datetime.now()
    now_us = now_kr - timedelta(hours=14)
    st.caption(f"🇰🇷 {now_kr.strftime('%y/%m/%d %H:%M')} | 🇺🇸 {now_us.strftime('%H:%M')} (NY)")

with c_b:
    b1, b2, b3 = st.columns(3)
    if b1.button("📖 매뉴얼"): show_manual()
    if b2.button("⚙️ 관리"): open_stock_manager()
    if b3.button("👤 로그아웃"): logout()

menu = st.radio("메뉴", ["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"], horizontal=True, label_visibility="collapsed")
st.divider()

# ==========================================
# 6. 탭별 상세 로직
# ==========================================

# [Tab 1] 자산
if menu == "📊 자산":
    total_ev, total_bv, data = 0, 0, []
    with st.spinner("Polygon 자산 동기화 중..."):
        for t in tickers:
            curr_p = fetch_current_price_polygon(t)
            q, a = my_portfolio[t]
            ev = curr_p * q; bv = a * q; profit = ev - bv
            pct = (profit / bv * 100) if bv > 0 else 0
            total_ev += ev; total_bv += bv
            data.append({"종목": f"{ticker_info[t][0]}({t})", "현재가": curr_p, "수익률": pct, "평가액": ev})
            time.sleep(0.2)
    
    t_profit = total_ev - total_bv
    t_pct = (t_profit / total_bv * 100) if total_bv > 0 else 0
    st.metric("총 자산 평가액", f"${total_ev:,.2f}", f"${t_profit:,.2f} ({t_pct:+.2f}%)")
    if data:
        st.dataframe(pd.DataFrame(data).sort_values("평가액", ascending=False), hide_index=True, use_container_width=True)

# [Tab 2] AI 예측 (GBR, SVR, 성공률, 투자 의견)
elif menu == "🔮 AI예측":
    st.warning("⚠️ 재미로만 참고해 주세요.")
    if tickers:
        c1, c2 = st.columns(2)
        sel = c1.selectbox("종목 선택", tickers)
        model_type = c2.selectbox("모델 선택", ["📈 Gradient Boosting", "🎯 SVR (비선형)", "📏 선형회귀"])
        
        if st.button("🤖 AI 정밀 분석 실행"):
            with st.spinner("분석 중..."):
                df_h = fetch_history_polygon(sel)
                if not df_h.empty and len(df_h) > 60:
                    # 백테스팅 (성공률 계산)
                    train_df = df_h.iloc[:-30]
                    actual_30 = df_h.iloc[-30:]['Close'].values
                    
                    def get_pred(data, days):
                        X = np.arange(len(data)).reshape(-1, 1)
                        y = data['Close'].values
                        if "Gradient" in model_type:
                            m = GradientBoostingRegressor(n_estimators=100).fit(X, y)
                        elif "SVR" in model_type:
                            m = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1e3)).fit(X, y)
                        else:
                            m = LinearRegression().fit(X, y)
                        return m.predict(np.arange(len(data), len(data)+days).reshape(-1, 1))

                    back_preds = get_pred(train_df, 30)
                    acc = 100 - (np.mean(np.abs((actual_30 - back_preds) / actual_30)) * 100)
                    
                    # 미래 예측
                    future_preds = get_pred(df_h, 30)
                    curr_p = df_h['Close'].iloc[-1]; pred_f = future_preds[-1]
                    pct = (pred_f - curr_p) / curr_p * 100

                    # 결과 및 의견 표시
                    res1, res2 = st.columns(2)
                    res1.metric("30일 뒤 예상", f"${pred_f:.2f}", f"{pct:+.2f}%")
                    res2.metric("모델 성공률", f"{acc:.1f}%")

                    st.divider()
                    if pct > 5 and acc > 85: st.success(f"🟢 **매수 권장**: 높은 신뢰도로 {pct:.1f}% 상승이 예상됩니다.")
                    elif pct < -5: st.error(f"🔴 **주의**: AI가 하락 흐름을 감지했습니다.")
                    else: st.info("⚪ **관망**: 뚜렷한 추세가 보이지 않습니다.")

                    # 차트
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df_h.index, df_h['Close'], color='gray', alpha=0.5)
                    fdates = [df_h.index[-1] + timedelta(days=i) for i in range(1, 31)]
                    ax.plot(fdates, future_preds, 'r-', linewidth=2)
                    st.pyplot(fig)
                    st.success("분석 완료!")
                else: st.error("데이터 부족")

# [Tab 3] 종합분석 (최적화 및 에러 방지 버전)
elif menu == "📉 종합분석":
    if not tickers:
        st.warning("분석할 종목이 없습니다.")
    else:
        sel_txt = st.selectbox("진단할 종목", [f"{ticker_info[t][0]} ({t})" for t in tickers])
        sel_ticker = sel_txt.split('(')[-1].replace(')', '')
        
        if st.button("🔍 상세 재무 진단 실행", use_container_width=True):
            with st.spinner("야후 서버에서 재무 데이터를 분석 중..."):
                # 1. 안전하게 데이터 가져오기
                info = fetch_safe_financials(sel_ticker)
                
                if not info:
                    st.error("현재 야후 서버 접속이 제한되었습니다. 잠시 후 다시 시도하거나 앱을 Reboot 해주세요.")
                else:
                    # 2. 변수 정의 (에러 방지의 핵심)
                    per = info.get('trailingPE')
                    pbr = info.get('priceToBook')
                    roe = info.get('returnOnEquity')
                    # biz_summary 변수를 여기서 명확히 정의해야 에러가 안 납니다.
                    biz_summary = info.get('longBusinessSummary', '설명이 없습니다.') 
                    
                    # 3. 주요 지표 표시 (한 번만 깔끔하게)
                    st.write(f"### 📊 {sel_ticker} 핵심 재무 지표")
                    c1, c2, c3 = st.columns(3)
                    
                    # 수치가 있을 때만 소수점 표시, 없으면 N/A
                    c1.metric("PER (주가수익비율)", f"{per:.2f}" if per else "N/A")
                    c2.metric("PBR (주가순자산비율)", f"{pbr:.2f}" if pbr else "N/A")
                    c3.metric("ROE (자기자본이익률)", f"{roe*100:.2f}%" if roe else "N/A")

                    # 4. AI 투자 의견 생성
                    st.divider()
                    score = 0
                    if per and 0 < per < 20: score += 1
                    if pbr and 0 < pbr < 1.5: score += 1
                    if roe and roe > 0.15: score += 1
                    
                    status = "🟢 투자 양호" if score >= 2 else ("🟡 보통" if score == 1 else "🔴 관망 권유")
                    st.subheader(f"🤖 AI 종합 진단 결과: {status}")
                    
                    # 5. 기업 개요 (접이식)
                    with st.expander("🏢 기업 상세 개요 보기"):
                        st.write(biz_summary)

                    # 6. 실적 차트 시각화
                    try:
                        # yfinance의 세션을 사용하여 안전하게 호출
                        ticker_obj = yf.Ticker(sel_ticker, session=get_safe_session())
                        fin = ticker_obj.quarterly_financials
                        if not fin.empty and 'Total Revenue' in fin.index:
                            st.write("### 📈 최근 분기 매출 추이")
                            # 데이터를 보기 좋게 전치(T)하여 막대 그래프 생성
                            rev_data = fin.loc['Total Revenue'].sort_index()
                            st.bar_chart(rev_data)
                        else:
                            st.caption("공시된 분기 실적 데이터가 없습니다.")
                    except Exception:
                        st.caption("서버 응답 지연으로 실적 차트를 불러올 수 없습니다.")

# [Tab 4] 스캔 (안전한 스캔)
elif menu == "📡 스캔":
    if st.button("🚀 Polygon 스캔 실행"):
        res = []
        with st.spinner("종목별 지표 계산 중 (분당 호출 제한 준수)..."):
            for t in tickers:
                df = fetch_history_polygon(t)
                if not df.empty and len(df) > 20:
                    c = df['Close']
                    pct = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100
                    diff = c.diff(); up = diff.clip(lower=0).rolling(14).mean(); down = -diff.clip(upper=0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (up / down).iloc[-1]))
                    sig = "🔥급등" if pct >= 3 else ("💎과매도" if rsi <= 30 else "")
                    res.append([t, f"{pct:+.2f}%", f"{rsi:.1f}", sig])
                time.sleep(1.2) # Polygon 무료플랜 분당 5회 제한 준수 핵심
            st.table(pd.DataFrame(res, columns=["티커", "등락", "RSI", "신호"]))

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
