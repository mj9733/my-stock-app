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
# 1. 기본 설정 및 스타일
# ==========================================
st.set_page_config(
    page_title="나만의 투자 비서",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 구글 시트 이름
SHEET_NAME = "stock_db"

def configure_fonts():
    if sys.platform == 'darwin': plt.rc('font', family='AppleGothic')
    elif sys.platform == 'win32': plt.rc('font', family='Malgun Gothic')
    else: plt.rc('font', family='NanumGothic') 
    plt.rcParams['axes.unicode_minus'] = False

configure_fonts()

# ==========================================
# 2. 구글 시트 핸들링 (Backend)
# ==========================================
@st.cache_resource
def get_google_sheet():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(SHEET_NAME)
        return sh.sheet1
    except Exception as e:
        st.error(f"구글 시트 연결 실패: {e}")
        return None

def load_portfolio_gs():
    sheet = get_google_sheet()
    if not sheet: return {}, {}
    try:
        data = sheet.get_all_records()
        if not data: return {}, {}
        
        my_portfolio = {}
        ticker_info = {}
        
        for row in data:
            # 헤더 대소문자/공백 유연하게 처리
            keys = {k.lower().strip(): k for k in row.keys()}
            t_key = keys.get('ticker')
            if not t_key: continue

            t = str(row[t_key]).strip().upper()
            if not t: continue
            
            # 값 가져오기 (컬럼명 대소문자 무시)
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
    except Exception as e: st.error(f"저장 실패: {e}")

# 데이터 로드
my_portfolio, ticker_info = load_portfolio_gs()

# ==========================================
# 3. 주가 데이터 (Market Data)
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
# 4. 사이드바 (사용자 메뉴)
# ==========================================
with st.sidebar:
    st.title("🚀 나만의 투자 비서")
    
    # 시계
    now = datetime.now(); ny_time = datetime.now() - timedelta(hours=14) 
    st.caption(f"🇰🇷 {now.strftime('%H:%M')} | 🇺🇸 {ny_time.strftime('%H:%M')}")
    st.divider()
    
    # 메뉴
    menu = st.radio("메뉴 선택", 
        ["📊 대시보드", "📡 전략 스캐너", "📉 리스크 분석", "📰 뉴스/심리", 
         "⚖️ 리밸런싱", "🔮 AI 예측", "🏢 펀더멘털"])
    
    st.divider()
    
    # [수정된 부분] 데이터 업로더 (누구나 자기 파일 사용 가능)
    with st.expander("📂 데이터 관리 (백업/복원)"):
        st.write("내 컴퓨터의 JSON 파일을 업로드하여 구글 시트에 저장합니다.")
        uploaded_file = st.file_uploader("JSON 파일 선택", type="json")
        
        if uploaded_file is not None:
            if st.button("📥 데이터 적용하기"):
                try:
                    local_db = json.load(uploaded_file)
                    l_port = local_db.get("portfolio", {})
                    l_info = local_db.get("info", {})
                    
                    if l_port:
                        save_portfolio_gs(l_port, l_info)
                        st.success("업로드 성공! 데이터가 갱신되었습니다.")
                        st.rerun()
                    else:
                        st.error("올바른 포트폴리오 파일이 아닙니다.")
                except Exception as e:
                    st.error(f"파일 읽기 오류: {e}")

    # 종목 직접 추가/삭제
    with st.expander("⚙️ 종목 직접 수정"):
        with st.form("add_stock"):
            t_in = st.text_input("티커 (예: TSLA)").upper()
            n_in = st.text_input("이름")
            q_in = st.number_input("수량", min_value=0)
            a_in = st.number_input("평단가", min_value=0.0)
            
            c1, c2 = st.columns(2)
            if c1.form_submit_button("💾 저장"):
                if t_in:
                    my_portfolio[t_in] = [q_in, a_in]
                    ticker_info[t_in] = [n_in if n_in else t_in, "-"]
                    save_portfolio_gs(my_portfolio, ticker_info)
                    st.success("저장 완료!")
                    st.rerun()
            
            if c2.form_submit_button("🗑️ 삭제"):
                if t_in in my_portfolio:
                    del my_portfolio[t_in]
                    if t_in in ticker_info: del ticker_info[t_in]
                    save_portfolio_gs(my_portfolio, ticker_info)
                    st.warning("삭제 완료!")
                    st.rerun()

# ==========================================
# 5. 메인 기능 (Dashboard & Features)
# ==========================================

# [Tab 1] 대시보드
if menu == "📊 대시보드":
    st.title("📊 자산 현황")
    
    # 매크로 지표
    c1, c2, c3, c4 = st.columns(4)
    macros = {"S&P500": "^GSPC", "나스닥": "^IXIC", "미국채10년": "^TNX", "달러": "DX-Y.NYB"}
    mp = fetch_all_prices(list(macros.values()))
    c1.metric("S&P 500", f"{mp['^GSPC']:,.0f}")
    c2.metric("나스닥", f"{mp['^IXIC']:,.0f}")
    c3.metric("미국채 10년", f"{mp['^TNX']:.2f}%")
    c4.metric("달러 인덱스", f"{mp['DX-Y.NYB']:.2f}")
    st.divider()
    
    # 자산 계산
    tb = 0; te = 0; data = []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        v = c * q; bv = a * q; p = v - bv
        pct = (p / bv * 100) if bv > 0 else 0
        tb += bv; te += v
        i = ticker_info.get(t, [t, "-"])
        data.append({"종목": i[0], "티커": t, "수량": q, "평단": a, "현재": c, "평가액": v, "수익률": pct, "수익금": p})
        
    c1, c2, c3 = st.columns(3)
    c1.metric("총 매수금액", f"${tb:,.0f}")
    c2.metric("총 평가금액", f"${te:,.0f}")
    c3.metric("총 수익금 (수익률)", f"${te-tb:+,.0f}", f"{(te-tb)/tb*100 if tb>0 else 0:+.2f}%")
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df.style.format({
            "평단": "${:,.2f}", "현재": "${:,.2f}", "평가액": "${:,.0f}", "수익금": "${:+,.0f}", "수익률": "{:+.2f}%"
        }), use_container_width=True, hide_index=True)
    else: st.info("데이터가 없습니다. 사이드바에서 파일을 업로드하거나 종목을 추가하세요.")

# [Tab 2] 전략 스캐너
elif menu == "📡 전략 스캐너":
    st.title("📡 시장 감시 & 전략 신호")
    if st.button("🚀 스캔 실행"):
        with st.spinner("시장 데이터를 분석 중입니다..."):
            try:
                df = yf.download(" ".join(tickers), period="2mo", progress=False)
                res = []
                for t in tickers:
                    try:
                        h = df.xs(t, level=1, axis=1) if len(tickers)>1 else df
                        if h.empty: continue
                        c = h['Close']; p = c.iloc[-1]; pct = (p - c.iloc[-2])/c.iloc[-2]*100
                        d = c.diff(); rsi = 100 - (100/(1 + d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0)).rolling(14).mean())).iloc[-1]
                        sig = "🔥 급등" if pct>=3 else ("💎 과매도" if rsi<=30 else ("⚠️ 과열" if rsi>=70 else "-"))
                        res.append([ticker_info[t][0], t, p, pct, rsi, sig])
                    except: pass
                st.dataframe(pd.DataFrame(res, columns=["종목","티커","현재가","등락","RSI","신호"]).style.format({"현재가":"${:.2f}", "등락":"{:.2f}%", "RSI":"{:.1f}"}), use_container_width=True, hide_index=True)
            except: st.error("데이터 오류 발생")

# [Tab 3] 리스크 분석 (Main과 동일하게 구현)
elif menu == "📉 리스크 분석":
    st.title("📉 리스크 & 변동성 분석")
    if st.button("🔍 정밀 분석 실행"):
        with st.spinner("1년치 데이터를 분석 중..."):
            try:
                df = yf.download(" ".join(tickers), period="1y", progress=False)['Close']
                res = []
                for t in tickers:
                    s = df[t] if len(tickers)>1 else df
                    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
                    vol = s.pct_change().std() * (252**0.5) * 100
                    res.append([ticker_info[t][0], mdd, vol])
                
                st.dataframe(pd.DataFrame(res, columns=["종목","MDD(최대낙폭)","변동성(연간)"]).style.format({"MDD(최대낙폭)":"{:.2f}%", "변동성(연간)":"{:.2f}%"}), use_container_width=True, hide_index=True)
                
                if len(tickers)>1:
                    st.subheader("🔗 상관관계 히트맵")
                    fig, ax = plt.subplots()
                    cax = ax.matshow(df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
                    fig.colorbar(cax)
                    names = [ticker_info[t][0] for t in tickers]
                    ax.set_xticks(range(len(tickers))); ax.set_yticks(range(len(tickers)))
                    ax.set_xticklabels(names, rotation=90, fontfamily='AppleGothic'); ax.set_yticklabels(names, fontfamily='AppleGothic')
                    st.pyplot(fig)
            except: st.error("분석 실패 (데이터 부족)")

# [Tab 4] 뉴스/심리 (Main 기능 완벽 이식)
elif menu == "📰 뉴스/심리":
    st.title("📰 글로벌 뉴스 AI 분석")
    if st.button("🌍 뉴스 가져오기"):
        with st.spinner("뉴스 수집 및 AI 분석 중..."):
            try: tr = GoogleTranslator(source='auto', target='ko')
            except: tr = None
            items = []; tot_score = 0
            pos = ['up','surge','gain','buy','high','bull','growth','profit','strong']; neg = ['down','drop','loss','sell','low','bear','crash','weak','debt']
            
            for t in tickers:
                try:
                    y = yf.Ticker(t); news = y.news
                    if not news: continue
                    n = news[0] # 최신 1개
                    
                    ttl = n.get('title') or "제목 없음"
                    link = n.get('link') or (n.get('clickThroughUrl',{}).get('url') if n.get('clickThroughUrl') else "")
                    
                    # 시간 파싱
                    time_s = "최근"
                    try: 
                        pt = n.get('providerPublishTime')
                        if pt: time_s = datetime.fromtimestamp(pt).strftime("%m-%d %H:%M")
                    except: pass
                    
                    # 번역
                    ko = ttl
                    if tr: 
                        try: ko = tr.translate(ttl)
                        except: pass
                        
                    # 감성 분석
                    sc = 0
                    for w in pos: 
                        if w in ttl.lower(): sc+=1
                    for w in neg: 
                        if w in ttl.lower(): sc-=1
                    tot_score += sc
                    
                    sent = "😊 호재" if sc>0 else ("😨 악재" if sc<0 else "😐 중립")
                    items.append({"시간":time_s, "종목":ticker_info[t][0], "감성":sent, "내용":ko, "링크":link})
                except: pass
            
            # 종합 결론 출력
            if items:
                msg = ""
                if tot_score >= 5: msg = f"🔥 종합: 강력 매수 신호 (불장) (+{tot_score}점)"; st.success(msg, icon="🔥")
                elif tot_score > 0: msg = f"😊 종합: 긍정적 흐름 (+{tot_score}점)"; st.success(msg, icon="😊")
                elif tot_score <= -5: msg = f"❄️ 종합: 폭락 주의 (패닉) ({tot_score}점)"; st.error(msg, icon="❄️")
                elif tot_score < 0: msg = f"😨 종합: 부정적 흐름 ({tot_score}점)"; st.warning(msg, icon="😨")
                else: msg = "😐 종합: 관망세 (중립) (0점)"; st.info(msg, icon="😐")

                st.dataframe(pd.DataFrame(items), column_config={"링크": st.column_config.LinkColumn("원문 보기")}, use_container_width=True, hide_index=True)
            else: st.warning("수집된 뉴스가 없습니다.")

# [Tab 5] 리밸런싱
elif menu == "⚖️ 리밸런싱":
    st.title("⚖️ 포트폴리오 리밸런싱")
    tv = sum([current_prices.get(t,0)*my_portfolio[t][0] for t in tickers])
    st.metric("총 자산 가치", f"${tv:,.0f}")
    
    df = pd.DataFrame([{"티커":t, "종목":ticker_info[t][0], "현재가":current_prices[t], "수량":my_portfolio[t][0], "비중": (current_prices[t]*my_portfolio[t][0]/tv*100) if tv>0 else 0, "목표":0.0} for t in tickers])
    
    st.caption("아래 표에서 '목표' 비중을 직접 입력하세요.")
    ed = st.data_editor(df, column_config={"목표":st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%.1f")}, use_container_width=True, hide_index=True)
    
    total_tgt = ed["목표"].sum()
    if 99.9 <= total_tgt <= 100.1:
        st.success(f"합계 {total_tgt:.1f}% 확인 완료! ✅")
        res = []
        t_buy=0; t_sell=0
        for i, r in ed.iterrows():
            diff = (tv * r['목표']/100) - (r['현재가']*r['수량'])
            q = int(diff / r['현재가']) if r['현재가']>0 else 0
            amt = q * r['현재가']
            if q > 0: t_buy += amt
            else: t_sell += abs(amt)
            res.append({"종목":r['종목'], "조정 수량":q, "매매 금액":amt})
        
        st.dataframe(pd.DataFrame(res).style.format({"매매 금액":"${:+,.0f}"}), use_container_width=True, hide_index=True)
        st.write(f"📉 총 매도: ${t_sell:,.0f}  |  📈 총 매수: ${t_buy:,.0f}")
    else:
        st.warning(f"현재 목표 합계: {total_tgt:.1f}% (100%를 맞춰주세요)")

# [Tab 6] AI 예측
elif menu == "🔮 AI 예측":
    st.title("🔮 AI 미래 가격 예측")
    sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]} ({t})" for t in tickers])
    sel = sel_txt.split('(')[-1].replace(')', '')
    
    if st.button("🤖 30일 뒤 예측 실행"):
        with st.spinner("AI가 학습 중입니다..."):
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
                col1, col2 = st.columns(2)
                col1.metric("현재 가격", f"${curr:.2f}")
                col2.metric("30일 뒤 예측", f"${pred:.2f}", f"{pct:+.2f}%")
                
                fig, ax = plt.subplots()
                ax.plot(df.index, df['Close'], label='Actual')
                ax.plot(df.index, model.predict(df[['D']]), '--', label='Trend')
                last_dt = df.index[-1]
                if isinstance(last_dt, pd.Timestamp):
                    fdates = [last_dt + timedelta(days=i) for i in range(1,31)]
                    ax.plot(fdates, model.predict(fut), 'r-', label='Future')
                ax.legend()
                st.pyplot(fig)
                
                if pct > 0: st.success("🚀 상승 추세가 예상됩니다.")
                else: st.error("📉 하락 추세가 우려됩니다.")
            except Exception as e: st.error(f"예측 실패: {e}")

# [Tab 7] 펀더멘털
elif menu == "🏢 펀더멘털":
    st.title("🏢 기업 펀더멘털 분석")
    sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]} ({t})" for t in tickers])
    sel = sel_txt.split('(')[-1].replace(')', '')
    
    if st.button("🔍 재무제표 분석"):
        t = yf.Ticker(sel); i = t.info
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("PER", f"{i.get('trailingPE',0):.2f}")
        c2.metric("PBR", f"{i.get('priceToBook',0):.2f}")
        c3.metric("ROE", f"{i.get('returnOnEquity',0)*100:.2f}%")
        c4.metric("매출성장", f"{i.get('revenueGrowth',0)*100:.2f}%")
        
        f = t.quarterly_financials
        if not f.empty:
            rev = f.loc['Total Revenue'][::-1] / 1e9
            net = f.loc['Net Income'][::-1] / 1e9
            fig, ax = plt.subplots()
            x = np.arange(len(rev)); w=0.35
            ax.bar(x-w/2, rev, w, label='매출($B)'); ax.bar(x+w/2, net, w, label='순이익($B)')
            ax.set_xticks(x); ax.set_xticklabels([d.strftime('%Y-%m') for d in rev.index])
            ax.legend(); st.pyplot(fig)