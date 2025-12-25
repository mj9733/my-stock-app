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
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="내 주식 비서 Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    except: return None

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
# 4. 상단 헤더 & 설정
# ==========================================
with st.sidebar:
    st.header("⚙️ 설정")
    with st.expander("📂 데이터 업로드"):
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
    
    with st.expander("✏️ 종목 관리"):
        with st.form("add_stock"):
            t_in = st.text_input("티커").upper()
            n_in = st.text_input("이름")
            q_in = st.number_input("수량", min_value=0)
            a_in = st.number_input("평단가", min_value=0.0)
            if st.form_submit_button("저장/삭제"):
                if t_in:
                    if t_in in my_portfolio and q_in == 0:
                        del my_portfolio[t_in]
                        if t_in in ticker_info: del ticker_info[t_in]
                    else:
                        my_portfolio[t_in] = [q_in, a_in]
                        ticker_info[t_in] = [n_in if n_in else t_in, "-"]
                    save_portfolio_gs(my_portfolio, ticker_info)
                    st.rerun()

# 메인 타이틀
col_title, col_clock = st.columns([2, 1])
now = datetime.now()
col_title.subheader("🚀 내 주식 비서")
col_clock.caption(f"🕒 {now.strftime('%H:%M')}")

# ==========================================
# 5. 메인 탭 메뉴
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"])

# [Tab 1] 자산
with tab1:
    macros = {"S&P500": "^GSPC", "나스닥": "^IXIC", "환율": "DX-Y.NYB"}
    mp = fetch_all_prices(list(macros.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("S&P500", f"{mp['^GSPC']:,.0f}")
    c2.metric("나스닥", f"{mp['^IXIC']:,.0f}")
    c3.metric("환율", f"{mp['DX-Y.NYB']:.1f}")
    st.divider()

    tb = 0; te = 0; data = []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        v = c * q; bv = a * q; p = v - bv
        pct = (p / bv * 100) if bv > 0 else 0
        tb += bv; te += v
        i = ticker_info.get(t, [t, "-"])
        data.append({"종목": i[0], "수량": q, "현재가": c, "평가액": v, "수익률": pct})

    tc1, tc2 = st.columns(2)
    tc1.metric("총 평가", f"${te:,.0f}")
    tc2.metric("총 수익", f"${te-tb:+,.0f}", f"{(te-tb)/tb*100 if tb>0 else 0:+.1f}%")

    if data:
        st.caption("👇 보유 종목 상세")
        st.dataframe(pd.DataFrame(data).style.format({"현재가":"${:,.0f}", "평가액":"${:,.0f}", "수익률":"{:+.1f}%"}), use_container_width=True, hide_index=True)
    else: st.info("종목을 추가해주세요.")

# [Tab 2] AI 예측 (요청하신 대로 명확하게!)
with tab2:
    sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]}" for t in tickers])
    sel = next((k for k, v in ticker_info.items() if v[0] == sel_txt), tickers[0])

    if st.button("🤖 30일 뒤 가격 예측 실행", use_container_width=True):
        with st.spinner("AI가 과거 데이터를 학습 중..."):
            try:
                # 1. 데이터 학습
                df = yf.download(sel, period="1y", progress=False)
                df = df[['Close']].dropna(); df['D'] = np.arange(len(df))
                model = LinearRegression().fit(df[['D']], df['Close'])
                
                # 2. 예측
                curr = df['Close'].iloc[-1]
                if hasattr(curr, 'item'): curr = curr.item()
                
                fut_days = np.arange(len(df), len(df)+30).reshape(-1,1)
                pred = model.predict(fut_days)[-1]
                if hasattr(pred, 'item'): pred = pred.item()
                
                pct = (pred - curr) / curr * 100
                
                # 3. 결과 표시 (나란히 비교)
                col1, col2 = st.columns(2)
                col1.metric("현재 가격", f"${curr:.2f}")
                col2.metric("30일 뒤 예상", f"${pred:.2f}", f"{pct:+.2f}%")
                
                # 4. 차트
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(df.index, df['Close'], label='과거 주가')
                ax.plot(df.index, model.predict(df[['D']]), '--', color='orange', label='추세선')
                
                # 미래 차트 연결
                last_dt = df.index[-1]
                future_dates = [last_dt + timedelta(days=i) for i in range(1, 31)]
                ax.plot(future_dates, model.predict(fut_days), 'r-', linewidth=2, label='예측 구간')
                
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
            except Exception as e: st.error(f"예측 실패: {e}")

# [Tab 3] 종합 분석 (리스크 + 펀더멘털 통합)
with tab3:
    st.write("📊 **리스크(MDD) & 가치평가(PER/PBR) 통합 분석**")
    if st.button("🔍 전체 종목 정밀 분석", use_container_width=True):
        with st.spinner("모든 종목의 재무제표와 차트를 분석 중입니다..."):
            try:
                # 데이터 다운로드 (1년치)
                df_chart = yf.download(" ".join(tickers), period="1y", progress=False)['Close']
                
                res = []
                for t in tickers:
                    # 1. 리스크 (MDD, 변동성)
                    s = df_chart[t] if len(tickers)>1 else df_chart
                    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
                    vol = s.pct_change().std() * (252**0.5) * 100
                    
                    # 2. 펀더멘털 (PER, PBR, ROE) - API 호출
                    try:
                        info = yf.Ticker(t).info
                        per = info.get('trailingPE', 0)
                        pbr = info.get('priceToBook', 0)
                        roe = info.get('returnOnEquity', 0)
                    except:
                        per = 0; pbr = 0; roe = 0
                    
                    res.append({
                        "종목": ticker_info[t][0],
                        "MDD(위험)": mdd,
                        "변동성": vol,
                        "PER": per if per else 0,
                        "PBR": pbr if pbr else 0,
                        "ROE": roe * 100 if roe else 0
                    })
                
                # 표 표시
                st.success("분석 완료!")
                st.dataframe(
                    pd.DataFrame(res),
                    column_config={
                        "MDD(위험)": st.column_config.NumberColumn(format="%.2f%%"),
                        "변동성": st.column_config.NumberColumn(format="%.2f%%"),
                        "PER": st.column_config.NumberColumn(format="%.2f배"),
                        "PBR": st.column_config.NumberColumn(format="%.2f배"),
                        "ROE": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("💡 팁: 표를 옆으로 밀어서 모든 데이터를 확인하세요.")
                
            except Exception as e: st.error(f"분석 중 오류: {e}")

# [Tab 4] 스캐너
with tab4:
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
                        
                        sig = ""
                        if pct>=3: sig = "🔥급등"
                        elif rsi<=30: sig = "💎과매도"
                        elif rsi>=70: sig = "⚠️과열"
                        
                        if sig: 
                            res.append([ticker_info[t][0], f"{pct:+.1f}%", f"{rsi:.0f}", sig])
                    except: pass
                
                if res:
                    st.dataframe(pd.DataFrame(res, columns=["종목","등락","RSI","신호"]), use_container_width=True, hide_index=True)
                else: st.info("특이사항 없음")
            except: st.error("데이터 오류")

# [Tab 5] 뉴스
with tab5:
    if st.button("🌍 뉴스 가져오기", use_container_width=True):
        with st.spinner("뉴스 분석 중..."):
            try: tr = GoogleTranslator(source='auto', target='ko')
            except: tr = None
            items = []; tot = 0
            pos = ['up','gain','buy','bull','strong']; neg = ['down','loss','sell','bear','weak']
            
            for t in tickers:
                try:
                    y = yf.Ticker(t); news = y.news
                    if not news: continue
                    n = news[0]; ttl = n.get('title') or ""
                    link = n.get('link') or ""
                    
                    ko = ttl
                    if tr: 
                        try: ko = tr.translate(ttl)
                        except: pass
                    
                    sc = 0
                    for w in pos: 
                        if w in ttl.lower(): sc+=1
                    for w in neg: 
                        if w in ttl.lower(): sc-=1
                    tot += sc
                    
                    sent = "😊" if sc>0 else ("😨" if sc<0 else "😐")
                    items.append({"감성":sent, "종목":ticker_info[t][0], "내용":ko, "링크":link})
                except: pass
            
            if items:
                msg = f"🔥 불장 (+{tot})" if tot>=3 else (f"❄️ 조심 ({tot})" if tot<=-3 else "😐 쏘쏘")
                st.info(msg)
                st.dataframe(pd.DataFrame(items), column_config={"링크": st.column_config.LinkColumn("원문")}, use_container_width=True, hide_index=True)
            else: st.warning("뉴스 없음")
