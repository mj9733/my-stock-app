import streamlit as st
import pandas as pd
import yfinance as yf
import json
import numpy as np
import matplotlib.pyplot as plt
import gspread
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from sklearn.linear_model import LinearRegression
import sys

# ==========================================
# 1. 기본 설정 (모바일 최적화)
# ==========================================
st.set_page_config(
    page_title="내 주식 비서",
    page_icon="📱",
    layout="wide", # 모바일에서도 꽉 차게
    initial_sidebar_state="collapsed" # 사이드바는 기본적으로 숨김
)

SHEET_NAME = "stock_db"

def configure_fonts():
    if sys.platform == 'darwin': plt.rc('font', family='AppleGothic')
    elif sys.platform == 'win32': plt.rc('font', family='Malgun Gothic')
    else: plt.rc('font', family='NanumGothic') 
    plt.rcParams['axes.unicode_minus'] = False

configure_fonts()

# ==========================================
# 2. 구글 시트 & 데이터 핸들링
# ==========================================
@st.cache_resource
def get_google_sheet():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(SHEET_NAME)
        return sh.sheet1
    except Exception as e: return None

def load_portfolio_gs():
    sheet = get_google_sheet()
    if not sheet: return {}, {}
    try:
        data = sheet.get_all_records()
        if not data: return {}, {}
        
        my_portfolio = {}
        ticker_info = {}
        
        for row in data:
            keys = {k.lower().strip(): k for k in row.keys()}
            t_key = keys.get('ticker')
            if not t_key: continue
            t = str(row[t_key]).strip().upper()
            if not t: continue
            
            qty = int(row.get(keys.get('qty', 'Qty'), 0) or 0)
            avg = float(row.get(keys.get('avg', 'Avg'), 0.0) or 0.0)
            name = str(row.get(keys.get('name', 'Name'), t))
            desc = str(row.get(keys.get('desc', 'Desc'), '-'))
            
            my_portfolio[t] = [qty, avg]
            ticker_info[t] = [name, desc]
        return my_portfolio, ticker_info
    except: return {}, {}

def save_portfolio_gs(my_portfolio, ticker_info):
    sheet = get_google_sheet()
    if not sheet: return
    try:
        rows = []
        for t, val in my_portfolio.items():
            qty, avg = val
            info = ticker_info.get(t, [t, "-"])
            rows.append([t, info[0], info[1], qty, avg])
        sheet.clear()
        sheet.append_row(["Ticker", "Name", "Desc", "Qty", "Avg"])
        if rows: sheet.append_rows(rows)
    except: pass

my_portfolio, ticker_info = load_portfolio_gs()

# ==========================================
# 3. 주가 데이터
# ==========================================
@st.cache_data(ttl=60)
def get_stock_price(ticker):
    try:
        t = yf.Ticker(ticker)
        p = t.fast_info.get('last_price', None)
        if p is None:
            hist = t.history(period="1d")
            if not hist.empty: p = hist['Close'].iloc[-1]
        return p if p else 0.0
    except: return 0.0

@st.cache_data(ttl=60) 
def fetch_all_prices(tickers):
    prices = {}
    for t in tickers: prices[t] = get_stock_price(t)
    return prices

tickers = list(my_portfolio.keys())
current_prices = fetch_all_prices(tickers)

# ==========================================
# 4. 상단 헤더 & 시계 (모바일용)
# ==========================================
# 사이드바는 '설정' 용도로만 사용 (평소에는 숨김)
with st.sidebar:
    st.header("⚙️ 설정 및 데이터")
    
    # 데이터 업로드
    with st.expander("📂 백업 데이터 업로드"):
        uploaded_file = st.file_uploader("JSON 파일", type="json")
        if uploaded_file is not None and st.button("적용하기"):
            try:
                local_db = json.load(uploaded_file)
                l_port = local_db.get("portfolio", {})
                l_info = local_db.get("info", {})
                if l_port:
                    save_portfolio_gs(l_port, l_info)
                    st.success("완료!")
                    st.rerun()
            except: st.error("오류")

    # 종목 수정
    with st.expander("✏️ 종목 수동 수정"):
        with st.form("add_stock"):
            t_in = st.text_input("티커").upper()
            n_in = st.text_input("이름")
            q_in = st.number_input("수량", min_value=0)
            a_in = st.number_input("평단가", min_value=0.0)
            if st.form_submit_button("저장/삭제"):
                if t_in:
                    if t_in in my_portfolio and q_in == 0: # 수량 0이면 삭제로 간주
                        del my_portfolio[t_in]
                        if t_in in ticker_info: del ticker_info[t_in]
                        st.warning(f"{t_in} 삭제됨")
                    else:
                        my_portfolio[t_in] = [q_in, a_in]
                        ticker_info[t_in] = [n_in if n_in else t_in, "-"]
                        st.success(f"{t_in} 저장됨")
                    save_portfolio_gs(my_portfolio, ticker_info)
                    st.rerun()

# 메인 타이틀 (시계 포함)
col_title, col_clock = st.columns([2, 1])
now = datetime.now()
col_title.subheader("🚀 내 주식 비서")
col_clock.caption(f"🕒 {now.strftime('%H:%M')}")

# ==========================================
# 5. 메인 탭 메뉴 (모바일 핵심 UI)
# ==========================================
# 메뉴를 상단 탭으로 변경 -> 터치하기 편함
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 자산", "📰 뉴스", "📡 스캔", "📉 리스크", "🔮 AI"])

# [Tab 1] 자산 (대시보드)
with tab1:
    # 1. 주요 지수 (작게 한줄로)
    macros = {"S&P500": "^GSPC", "나스닥": "^IXIC", "환율": "DX-Y.NYB"}
    mp = fetch_all_prices(list(macros.values()))
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("S&P500", f"{mp['^GSPC']:,.0f}")
    mc2.metric("나스닥", f"{mp['^IXIC']:,.0f}")
    mc3.metric("달러", f"{mp['DX-Y.NYB']:.1f}")
    
    st.divider()

    # 2. 내 자산 요약
    tb = 0; te = 0; data = []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        v = c * q; bv = a * q; p = v - bv
        pct = (p / bv * 100) if bv > 0 else 0
        tb += bv; te += v
        i = ticker_info.get(t, [t, "-"])
        data.append({"종목": i[0], "티커": t, "수량": q, "평단": a, "현재": c, "평가": v, "수익률": pct, "수익금": p})

    tc1, tc2 = st.columns(2)
    tc1.metric("총 평가금", f"${te:,.0f}")
    tc2.metric("총 수익", f"${te-tb:+,.0f}", f"{(te-tb)/tb*100 if tb>0 else 0:+.1f}%")

    # 3. 보유 종목 리스트 (모바일용 간소화)
    if data:
        st.caption("👇 보유 종목 상세")
        df = pd.DataFrame(data)
        # 모바일에서는 컬럼이 많으면 안보임. 핵심만 보여주기
        st.dataframe(
            df[["종목", "현재", "수익률", "평가"]], 
            column_config={
                "현재": st.column_config.NumberColumn(format="$%.0f"),
                "평가": st.column_config.NumberColumn(format="$%.0f"),
                "수익률": st.column_config.NumberColumn(format="%.1f%%")
            },
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("데이터 없음. 사이드바(>)에서 추가하세요.")

# [Tab 2] 뉴스 (모바일 최적화)
with tab2:
    if st.button("🌍 뉴스 새로고침", use_container_width=True):
        with st.spinner("분석 중..."):
            try: tr = GoogleTranslator(source='auto', target='ko')
            except: tr = None
            items = []; tot_score = 0
            pos = ['up','surge','gain','buy','bull','strong']; neg = ['down','drop','loss','sell','bear','weak']
            
            for t in tickers:
                try:
                    y = yf.Ticker(t); news = y.news
                    if not news: continue
                    n = news[0]
                    ttl = n.get('title') or ""
                    link = n.get('link') or (n.get('clickThroughUrl',{}).get('url') if n.get('clickThroughUrl') else "")
                    
                    ko = ttl
                    if tr: 
                        try: ko = tr.translate(ttl)
                        except: pass
                    
                    sc = 0
                    for w in pos: 
                        if w in ttl.lower(): sc+=1
                    for w in neg: 
                        if w in ttl.lower(): sc-=1
                    tot_score += sc
                    
                    sent = "😊" if sc>0 else ("😨" if sc<0 else "😐")
                    items.append({"감성":sent, "종목":ticker_info[t][0], "내용":ko, "링크":link})
                except: pass
            
            if items:
                msg = f"🔥 불장 (+{tot_score})" if tot_score>=3 else (f"❄️ 조심 ({tot_score})" if tot_score<=-3 else "😐 쏘쏘")
                st.info(msg)
                st.dataframe(
                    pd.DataFrame(items), 
                    column_config={"링크": st.column_config.LinkColumn("원문")}, 
                    use_container_width=True, 
                    hide_index=True
                )
            else: st.warning("뉴스 없음")

# [Tab 3] 전략 스캐너
with tab3:
    if st.button("🚀 급등/과매도 스캔", use_container_width=True):
        with st.spinner("스캔 중..."):
            try:
                df = yf.download(" ".join(tickers), period="2mo", progress=False)
                res = []
                for t in tickers:
                    try:
                        h = df.xs(t, level=1, axis=1) if len(tickers)>1 else df
                        if h.empty: continue
                        c = h['Close']; p = c.iloc[-1]; pct = (p - c.iloc[-2])/c.iloc[-2]*100
                        d = c.diff(); rsi = 100 - (100/(1 + d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0)).rolling(14).mean())).iloc[-1]
                        
                        # 특징적인 것만 리스트업
                        sig = ""
                        if pct>=3: sig = "🔥급등"
                        elif rsi<=30: sig = "💎줍줍"
                        elif rsi>=70: sig = "⚠️과열"
                        
                        if sig: # 신호 있는것만 보여주기 (모바일 공간 절약)
                            res.append([ticker_info[t][0], f"{pct:+.1f}%", f"{rsi:.0f}", sig])
                    except: pass
                
                if res:
                    st.dataframe(pd.DataFrame(res, columns=["종목","등락","RSI","신호"]), use_container_width=True, hide_index=True)
                else:
                    st.info("특이사항 있는 종목이 없습니다.")
            except: st.error("데이터 오류")

# [Tab 4] 리스크
with tab4:
    if st.button("📉 변동성 분석", use_container_width=True):
        with st.spinner("분석 중..."):
            try:
                df = yf.download(" ".join(tickers), period="1y", progress=False)['Close']
                res = []
                for t in tickers:
                    s = df[t] if len(tickers)>1 else df
                    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
                    res.append([ticker_info[t][0], mdd])
                
                # 차트로 보여주기 (모바일은 표보다 차트가 나음)
                st.caption("최대 낙폭 (MDD)")
                st.bar_chart(pd.DataFrame(res, columns=["종목","MDD"]).set_index("종목"))
            except: st.error("실패")

# [Tab 5] AI 예측
with tab5:
    sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]}" for t in tickers])
    # 티커 찾기
    sel = next((k for k, v in ticker_info.items() if v[0] == sel_txt), tickers[0])

    if st.button("🤖 30일 뒤 예측", use_container_width=True):
        with st.spinner("AI 계산 중..."):
            try:
                df = yf.download(sel, period="1y", progress=False)
                df = df[['Close']].dropna(); df['D'] = np.arange(len(df))
                model = LinearRegression().fit(df[['D']], df['Close'])
                fut = np.arange(len(df), len(df)+30).reshape(-1,1)
                pred = model.predict(fut)[-1]
                curr = df['Close'].iloc[-1]
                if hasattr(curr, 'item'): curr = curr.item()
                if hasattr(pred, 'item'): pred = pred.item()
                
                pct = (pred-curr)/curr*100
                st.metric("30일 후 예상가", f"${pred:.2f}", f"{pct:+.1f}%")
                
                # 차트 그리기
                fig, ax = plt.subplots(figsize=(4, 3)) # 모바일용 작은 사이즈
                ax.plot(df.index, df['Close'], label='현재')
                ax.plot(df.index, model.predict(df[['D']]), '--', label='추세')
                st.pyplot(fig)
            except: st.error("실패")
