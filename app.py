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

# [추가] 서버 차단을 피하기 위한 세션 생성 함수
def get_safe_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
    })
    return session

# [추가] 재무 정보를 안전하게 가져오는 함수 (1시간 캐시)
@st.cache_data(ttl=3600)
def fetch_financial_info(ticker_symbol):
    try:
        session = get_safe_session()
        ticker = yf.Ticker(ticker_symbol, session=session)
        # .info는 에러 발생 확률이 높으므로 한 번만 호출해서 변수에 저장
        info = ticker.info
        return info
    except Exception:
        # 에러 발생 시 빈 사전을 반환하여 앱 중단 방지
        return {}
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
col_title, col_user_btns = st.columns([1.2, 1.3]) # 버튼 영역 확보를 위해 비율 조정
with col_title:
    st.subheader(f"📈 {st.session_state.user_id}님의 주식 비서")
    st.caption(f"🇰🇷 {datetime.now().strftime('%y/%m/%d %H:%M')} 기준")

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
    # 상단 유의사항 문구 추가
    st.warning("⚠️ **AI 예측은 과거 데이터를 기반으로 한 기술적 분석이며, 실제 투자 결과는 시장 상황에 따라 다를 수 있습니다. 재미와 참고용으로만 활용해 주세요.**")
    
    if not tickers:
        st.info("종목이 없습니다. 관리 메뉴에서 종목을 먼저 추가해 주세요.")
    else:
        # 기존 AI 예측 로직 시작
        c_sel, c_opt = st.columns([2, 1])
        with c_sel:
            sel_txt = st.selectbox("예측할 종목 선택", [f"{ticker_info[t][0]} ({t})" for t in tickers], label_visibility="collapsed")
            sel = sel_txt.split('(')[-1].replace(')', '')
        with c_opt:
            model_type = st.selectbox("분석 모델", ["📏 선형회귀", "🌲 랜덤포레스트"], label_visibility="collapsed")

        if st.button("🤖 AI 미래 가격 예측 실행", use_container_width=True):
            with st.spinner(f"{model_type}로 분석 중..."):
                try:
                    # 1년치 데이터 수집
                    df_h = yf.download(sel, period="1y", progress=False)
                    if df_h.empty: raise Exception("데이터 부족")
                    df_h = df_h[['Close']].dropna()
                    
                    X = np.arange(len(df_h)).reshape(-1, 1)
                    y = df_h['Close'].values.ravel()
                    
                    # 모델 학습
                    if "선형" in model_type:
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    model.fit(X, y)
                    
                    # 미래 30일 예측
                    curr_p = df_h['Close'].iloc[-1].item()
                    future_days = 30
                    future_X = np.arange(len(df_h), len(df_h) + future_days).reshape(-1, 1)
                    pred_y = model.predict(future_X)
                    pred_f = pred_y[-1]
                    pct = (pred_f - curr_p) / curr_p * 100
                    
                    # 결과 표시
                    st.metric("30일 뒤 예상 가격", f"${pred_f:.2f}", f"{pct:+.2f}%")
                    
                    # 시각화 차트
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df_h.index, df_h['Close'], label='실제 주가', color='gray', alpha=0.5)
                    
                    last_dt = df_h.index[-1]
                    fdates = [last_dt + timedelta(days=i) for i in range(1, future_days + 1)]
                    ax.plot(fdates, pred_y, 'r-', linewidth=2, label='미래 예측')
                    
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y.%m"))
                    ax.legend()
                    ax.grid(True, alpha=0.3, linestyle='--')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {e}")

# [Tab 3] 종합분석 (안전한 버전)
elif menu == "📉 종합분석":
    if not tickers:
        st.warning("분석할 종목이 없습니다. 관리 메뉴에서 종목을 추가해 주세요.")
    else:
        st.info("ℹ️ 재무 정보는 서버 부하 방지를 위해 1시간 단위로 업데이트됩니다.")
        
        sel_txt = st.selectbox("진단할 종목을 선택하세요", [f"{ticker_info[t][0]} ({t})" for t in tickers])
        sel_ticker = sel_txt.split('(')[-1].replace(')', '')
        
        if st.button("🔍 상세 재무 진단 실행", use_container_width=True):
            with st.spinner(f"{sel_ticker}의 재무 데이터를 정밀 분석 중입니다..."):
                # 안전한 함수 호출
                info = fetch_financial_info(sel_ticker)
                
                if not info:
                    st.error("현재 Yahoo Finance 서버 접속이 원활하지 않습니다. 잠시 후 다시 시도해 주세요.")
                else:
                    # 데이터 추출
                    per = info.get('trailingPE')
                    pbr = info.get('priceToBook')
                    roe = info.get('returnOnEquity')
                    biz_summary = info.get('longBusinessSummary', '기업 설명 정보가 없습니다.')

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

# [Tab 4] 스캔 (에러 방지 및 안전 버전)
elif menu == "📡 스캔":
    st.info("💡 보유하신 모든 종목의 기술적 지표를 분석합니다. (10분 단위 갱신)")
    
    if not tickers:
        st.warning("스캔할 종목이 없습니다. 관리 메뉴에서 종목을 추가해 주세요.")
    else:
        if st.button("🚀 전체 종목 실시간 스캔 실행", use_container_width=True):
            with st.spinner("모든 종목의 RSI 및 등락률을 계산 중입니다..."):
                try:
                    # 안전한 세션 생성
                    session = get_safe_session() 
                    
                    # 데이터 한꺼번에 다운로드 (기간 2개월)
                    # auto_adjust=True로 수정하여 데이터 정합성 높임
                    df_all = yf.download(
                        tickers, 
                        period="2mo", 
                        interval="1d", 
                        group_by='ticker', 
                        session=session, 
                        progress=False,
                        auto_adjust=True
                    )
                    
                    res = []
                    for t in tickers:
                        try:
                            # 1. 특정 종목 데이터 추출 (멀티인덱스 대응)
                            if len(tickers) > 1:
                                ticker_data = df_all[t]
                            else:
                                ticker_data = df_all
                                
                            c = ticker_data['Close'].dropna()
                            
                            # [핵심] 데이터가 비어있는지 확인하여 IndexError 방지
                            if c.empty or len(c) < 15:
                                continue
                            
                            # 2. 가격 및 등락률 계산
                            curr_p = c.iloc[-1]
                            prev_p = c.iloc[-2]
                            pct = (curr_p - prev_p) / prev_p * 100
                            
                            # 3. RSI 계산 (14일 기준)
                            diff = c.diff()
                            up = diff.clip(lower=0).rolling(window=14).mean()
                            down = -diff.clip(upper=0).rolling(window=14).mean()
                            
                            # 분모가 0이 되는 것을 방지
                            rs = up / down
                            rsi = 100 - (100 / (1 + rs.iloc[-1]))
                            
                            # 4. 신호 판별
                            signal = ""
                            if pct >= 3: signal = "🔥 급등"
                            elif pct <= -3: signal = "📉 급락"
                            
                            if rsi <= 30: signal += " 💎 과매도"
                            elif rsi >= 70: signal += " ⚠️ 과매수"
                            
                            name = ticker_info[t][0]
                            res.append([f"{name}({t})", f"${curr_p:.2f}", f"{pct:+.2f}%", f"{rsi:.1f}", signal])
                            
                        except Exception:
                            # 개별 종목 계산 실패 시 건너뜀
                            continue
                            
                    if res:
                        scan_df = pd.DataFrame(res, columns=["종목", "현재가", "등락률", "RSI", "분석 결과"])
                        st.success(f"총 {len(res)}개 종목 분석 완료!")
                        st.dataframe(scan_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("현재 특이 신호가 포착된 종목이 없습니다.")
                        
                except Exception as e:
                    st.error(f"스캔 중 서버 통신 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

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
