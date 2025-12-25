import streamlit as st
import pandas as pd
import yfinance as yf
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gspread
import os
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import sys
# [New] 자동 새로고침 라이브러리
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="내 주식 비서 Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [핵심] 1분(60초)마다 화면 자동 새로고침 (60 * 1000 밀리초)
st_autorefresh(interval=60 * 1000, key="data_refresh")

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
# [수정] ttl=50 (자동 새로고침 60초보다 짧게 설정해서 무조건 새 데이터 가져오게 함)
@st.cache_data(ttl=50)
def get_stock_price(ticker):
    try:
        t = yf.Ticker(ticker)
        p = t.fast_info.get('last_price', None)
        if p is None:
            hist = t.history(period="1d")
            if not hist.empty: p = hist['Close'].iloc[-1]
        return p if p else 0.0
    except: return 0.0

@st.cache_data(ttl=50) 
def fetch_all_prices(tickers):
    prices = {}
    for t in tickers: prices[t] = get_stock_price(t)
    return prices

tickers = list(my_portfolio.keys())
current_prices = fetch_all_prices(tickers)

# ==========================================
# 4. 팝업창들
# ==========================================
@st.dialog("📖 앱 사용 가이드")
def show_guide():
    st.write("### 탭별 기능 설명")
    st.markdown("""
    1. **📊 자산:** 수익률 순서대로 정렬해서 보고, **빨강(수익)/파랑(손실)**으로 상태를 확인하세요.
    2. **🔮 AI예측:** 과거 데이터를 기반으로 30일 뒤 주가를 예측합니다.
    3. **📉 종합분석:** 기업 재무를 분석해 긍정/부정 요인을 알려줍니다.
    4. **📡 스캔:** '급등'하거나 '과매도'된 종목을 포착합니다.
    5. **📰 뉴스:** 한국 관련 뉴스를 분석해 호재/악재를 파악합니다.
    """)
    st.caption("ℹ️ 이 앱은 1분마다 자동으로 데이터를 갱신합니다.")

@st.dialog("📋 종목 관리 (Excel 방식)")
def open_stock_manager():
    st.caption("아래 표를 클릭해서 종목을 자유롭게 추가/수정/삭제하세요.")
    rows = []
    for t in my_portfolio:
        qty, avg = my_portfolio[t]
        name, desc = ticker_info.get(t, [t, "-"])
        rows.append({"Ticker": t, "Name": name, "Qty": qty, "Avg": avg})
    
    df_current = pd.DataFrame(rows)
    if df_current.empty: df_current = pd.DataFrame(columns=["Ticker", "Name", "Qty", "Avg"])

    edited_df = st.data_editor(
        df_current, num_rows="dynamic", use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("티커 (예: TSLA)", required=True),
            "Name": st.column_config.TextColumn("이름 (예: 테슬라)", required=True),
            "Qty": st.column_config.NumberColumn("수량", min_value=1, step=1, required=True),
            "Avg": st.column_config.NumberColumn("평단가($)", min_value=0.0, step=0.01, required=True, format="%.2f"),
        }, hide_index=True
    )

    if st.button("💾 변경사항 저장하기", use_container_width=True):
        new_portfolio = {}
        new_info = {}
        for index, row in edited_df.iterrows():
            t = str(row["Ticker"]).strip().upper()
            n = str(row["Name"]).strip()
            q = int(row["Qty"])
            a = float(row["Avg"])
            if t:
                new_portfolio[t] = [q, a]
                new_info[t] = [n, "-"]
        save_portfolio_gs(new_portfolio, new_info)
        st.success("저장되었습니다! (새로고침 중...)")
        st.rerun()

# ==========================================
# 5. 메인 UI
# ==========================================
col_title, col_btns = st.columns([1.5, 1])
now_kr = datetime.now()
now_us = now_kr - timedelta(hours=14)

with col_title:
    st.subheader("🚀 내 주식 비서")
    st.caption(f"🇰🇷 {now_kr.strftime('%m/%d %H:%M')}  |  🇺🇸 {now_us.strftime('%H:%M')} (NY)")

with col_btns:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("❓ 가이드", use_container_width=True):
            show_guide()
    with b2:
        if st.button("⚙️ 종목 관리", use_container_width=True):
            open_stock_manager()

selected_menu = st.radio(
    "메뉴 이동",
    ["📊 자산", "🔮 AI예측", "📉 종합분석", "📡 스캔", "📰 뉴스"],
    horizontal=True, label_visibility="collapsed"
)
st.divider()

# [Tab 1] 자산
if selected_menu == "📊 자산":
    macros = {"S&P500": "^GSPC", "나스닥": "^IXIC", "달러 지표": "DX-Y.NYB"}
    mp = fetch_all_prices(list(macros.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("S&P500", f"{mp['^GSPC']:,.2f}")
    c2.metric("나스닥", f"{mp['^IXIC']:,.2f}")
    c3.metric("달러 지표", f"{mp['DX-Y.NYB']:,.2f}")
    st.divider()

    tb = 0; te = 0; data = []
    for t in tickers:
        q, a = my_portfolio[t]; c = current_prices.get(t, 0)
        v = c * q; bv = a * q; p = v - bv
        pct = (p / bv * 100) if bv > 0 else 0
        tb += bv; te += v
        i = ticker_info.get(t, [t, "-"])
        data.append({"종목": i[0], "수량": q, "현재가": c, "평가액": v, "평가손익": p, "수익률": pct})

    tc1, tc2 = st.columns(2)
    tc1.metric("총 평가", f"${te:,.2f}")
    
    # 총 수익 HTML 커스텀
    total_profit = te - tb
    profit_pct = (total_profit / tb * 100) if tb > 0 else 0
    
    if total_profit > 0:
        p_color = "#d62728"; arrow = "▲"; sign = "+"
    elif total_profit < 0:
        p_color = "#1f77b4"; arrow = "▼"; sign = ""
    else:
        p_color = "gray"; arrow = "-"; sign = ""
    
    tc2.markdown(f"""
        <div style="padding: 0px;">
            <p style="font-size: 14px; margin-bottom: 0px; opacity: 0.6;">총 수익</p>
            <p style="font-size: 30px; font-weight: bold; margin: 0px;">${total_profit:+,.2f}</p>
            <p style="font-size: 16px; font-weight: bold; color: {p_color}; margin: 0px;">
                {arrow} {sign}{profit_pct:.2f}%
            </p>
        </div>
    """, unsafe_allow_html=True)

    if data:
        st.caption("👇 보유 종목 상세")
        sort_opt = st.radio(
            "정렬 기준:",
            ["평가액순 (기본)", "수익률 높은순 🔥", "수익률 낮은순 💧", "현재가 높은순 💰"],
            horizontal=True, label_visibility="collapsed"
        )
        
        df = pd.DataFrame(data)
        if "수익률 높은순" in sort_opt: df = df.sort_values("수익률", ascending=False)
        elif "수익률 낮은순" in sort_opt: df = df.sort_values("수익률", ascending=True)
        elif "현재가" in sort_opt: df = df.sort_values("현재가", ascending=False)
        else: df = df.sort_values("평가액", ascending=False)

        def color_profit(val):
            return 'color: #d62728; font-weight: bold;' if val > 0 else ('color: #1f77b4; font-weight: bold;' if val < 0 else 'color: black')

        def format_arrow(val):
            if val > 0: return f"▲ {val:.2f}%"
            elif val < 0: return f"▼ {abs(val):.2f}%"
            return f"- {val:.2f}%"
        
        def format_money(val): return f"${val:,.2f}"

        st.dataframe(
            df.style.map(color_profit, subset=['수익률', '평가손익']).format({
                '수익률': format_arrow, '현재가': format_money, '평가액': format_money, '평가손익': lambda x: f"${x:+,.2f}"
            }),
            hide_index=True, use_container_width=True
        )
    else: st.info("👆 위 '⚙️ 종목 관리' 버튼을 눌러 종목을 추가하세요!")

# [Tab 2] AI 예측
elif selected_menu == "🔮 AI예측":
    if not tickers: st.warning("종목이 없습니다.")
    else:
        sel_txt = st.selectbox("종목 선택", [f"{ticker_info[t][0]}" for t in tickers])
        sel = next((k for k, v in ticker_info.items() if v[0] == sel_txt), tickers[0])

        if st.button("🤖 30일 뒤 가격 예측 실행", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                try:
                    df = yf.download(sel, period="1y", progress=False)
                    if df.empty: raise Exception("데이터 부족")
                    df = df[['Close']].dropna(); df['D'] = np.arange(len(df))
                    model = LinearRegression().fit(df[['D']], df['Close'])
                    curr = df['Close'].iloc[-1]
                    if hasattr(curr, 'item'): curr = curr.item()
                    fut_days = np.arange(len(df), len(df)+30).reshape(-1,1)
                    pred = model.predict(fut_days)[-1]
                    if hasattr(pred, 'item'): pred = pred.item()
                    pct = (pred - curr) / curr * 100
                    col1, col2 = st.columns(2)
                    col1.metric("현재 가격", f"${curr:.2f}")
                    col2.metric("30일 뒤 예상", f"${pred:.2f}", f"{pct:+.2f}%")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df.index, df['Close'], label='과거')
                    ax.plot(df.index, model.predict(df[['D']]), '--', color='orange', label='추세')
                    last_dt = df.index[-1]
                    fdates = [last_dt + timedelta(days=i) for i in range(1, 31)]
                    ax.plot(fdates, model.predict(fut_days), 'r-', linewidth=2, label='예측')
                    ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                except: st.error("데이터 부족")

# [Tab 3] 종합 분석
elif selected_menu == "📉 종합분석":
    st.write("📊 **기업 펀더멘털 & 리스크 정밀 진단**")
    if not tickers: st.warning("종목이 없습니다.")
    else:
        sel_fund = st.selectbox("분석할 종목을 선택하세요", [f"{ticker_info[t][0]} ({t})" for t in tickers])
        sel_ticker = sel_fund.split('(')[-1].replace(')', '')
        
        if st.button("🔍 상세 분석 및 진단", use_container_width=True):
            with st.spinner(f"{sel_ticker} 정밀 진단 중..."):
                try:
                    t = yf.Ticker(sel_ticker)
                    info = t.info
                    metrics = {
                        "시가총액": info.get("marketCap", 0), "현재가": info.get("currentPrice", 0),
                        "PER": info.get("trailingPE", 0), "PBR": info.get("priceToBook", 0),
                        "ROE": info.get("returnOnEquity", 0), "부채비율": info.get("debtToEquity", 0)
                    }

                    def get_status(k, v):
                        if not v: return None
                        if k == "PER": return "✅ 저평가" if 0 < v < 20 else ("⚠️ 고평가" if v > 50 else None)
                        if k == "PBR": return "✅ 저PBR" if 0 < v < 1.5 else ("⚠️ 고PBR" if v > 5 else None)
                        if k == "ROE": return "👑 고수익" if v > 0.15 else ("📉 수익저조" if v < 0.05 else None)
                        if k == "부채비율": return "🛡️ 건전" if v < 100 else ("🚨 위험" if v > 200 else None)
                        return None

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("현재가", f"${metrics['현재가']:,.2f}")
                    c2.metric("시가총액", f"${metrics['시가총액']/1e9:,.2f} B")
                    c3.metric("PER", f"{metrics['PER']:.2f}" if metrics['PER'] else "-", get_status("PER", metrics['PER']))
                    c4.metric("PBR", f"{metrics['PBR']:.2f}" if metrics['PBR'] else "-", get_status("PBR", metrics['PBR']))
                    c5, c6 = st.columns(2)
                    c5.metric("ROE (자기자본이익률)", f"{metrics['ROE']*100:.2f}%" if metrics['ROE'] else "-", get_status("ROE", metrics['ROE']))
                    c6.metric("부채비율", f"{metrics['부채비율']:.2f}%" if metrics['부채비율'] else "-", get_status("부채비율", metrics['부채비율']))
                    st.divider()

                    score = 0; reasons = []; warns = []
                    if metrics['PER']:
                        if 0 < metrics['PER'] < 20: score += 1; reasons.append("저평가 매력")
                        elif metrics['PER'] > 50: score -= 1; warns.append("고평가 부담")
                    if metrics['PBR']:
                        if 0 < metrics['PBR'] < 1.5: score += 1; reasons.append("자산가치 우수")
                        elif metrics['PBR'] > 5: score -= 1; warns.append("주가 과열")
                    if metrics['ROE']:
                        if metrics['ROE'] > 0.15: score += 1; reasons.append("높은 수익성")
                        elif metrics['ROE'] < 0.05: score -= 1; warns.append("이익률 저조")
                    if metrics['부채비율']:
                        if metrics['부채비율'] < 100: score += 1; reasons.append("재무 튼튼")
                        elif metrics['부채비율'] > 200: score -= 1; warns.append("부채 과다")

                    eval_msg = ""
                    if score >= 2: eval_msg = "🟢 **강력 매수 고려 (매우 우량함)**"
                    elif score == 1: eval_msg = "🟡 **매수 고려 (양호함)**"
                    elif score == 0: eval_msg = "⚪ **관망 (장단점 혼재)**"
                    else: eval_msg = "🔴 **투자 주의 (리스크 관리 필요)**"

                    st.subheader(f"종합평가: {eval_msg}")
                    if reasons: st.caption(f"👍 장점: {', '.join(reasons)}")
                    if warns: st.caption(f"👎 단점: {', '.join(warns)}")

                    fin = t.quarterly_financials
                    if not fin.empty:
                        rev = fin.loc['Total Revenue'][::-1] / 1e9
                        net = fin.loc['Net Income'][::-1] / 1e9
                        dates = [d.strftime('%Y-%m') for d in rev.index]
                        fig, ax = plt.subplots(figsize=(6, 3))
                        x = np.arange(len(dates)); width = 0.35
                        ax.bar(x - width/2, rev, width, label='매출 ($B)', color='#1f77b4', alpha=0.7)
                        ax.bar(x + width/2, net, width, label='순이익 ($B)', color='#2ca02c', alpha=0.7)
                        ax.set_xticks(x); ax.set_xticklabels(dates)
                        ax.legend(); ax.set_title("최근 분기 실적")
                        st.pyplot(fig)
                except Exception as e: st.error(f"오류: {e}")

# [Tab 4] 스캐너
elif selected_menu == "📡 스캔":
    if st.button("🚀 스캔 시작", use_container_width=True):
        with st.spinner("스캔 중..."):
            try:
                if not tickers: raise Exception("종목 없음")
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
                        if sig: res.append([ticker_info[t][0], f"{pct:+.2f}%", f"{rsi:.2f}", sig])
                    except: pass
                if res: st.dataframe(pd.DataFrame(res, columns=["종목","등락","RSI","신호"]), hide_index=True, use_container_width=True)
                else: st.info("특이사항 없음")
            except: st.error("종목을 먼저 추가해주세요.")

# [Tab 5] 뉴스
elif selected_menu == "📰 뉴스":
    if st.button("🌍 뉴스 분석", use_container_width=True):
        with st.spinner("뉴스 분석 중..."):
            items = []
            total_score = 0
            pos_words = ['상승', '급등', '최고', '호재', '매수', '수익', '기대', '강세', '돌파', '개선', '성장', '대박', '폭등']
            neg_words = ['하락', '급락', '최저', '악재', '매도', '손실', '우려', '약세', '붕괴', '감소', '위기', '폭락']

            for t in tickers:
                try:
                    query = f"{ticker_info[t][0]} {t}"
                    encoded_query = urllib.parse.quote(query)
                    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
                    feed = feedparser.parse(rss_url)
                    if feed.entries:
                        entry = feed.entries[0]
                        title = entry.title
                        link = entry.link
                        score = 0
                        for w in pos_words:
                            if w in title: score += 1
                        for w in neg_words:
                            if w in title: score -= 1
                        total_score += score
                        sent = "😐 중립"
                        if score > 0: sent = "😊 긍정"
                        elif score < 0: sent = "😨 부정"
                        items.append({"감성": sent, "종목": ticker_info[t][0], "제목": title, "링크": link})
                except: pass
            
            if items:
                msg = ""
                if total_score >= 3: msg = f"🔥 종합: 강력 매수 신호 (불장) (+{total_score})"
                elif total_score > 0: msg = f"😊 종합: 긍정적 흐름 (+{total_score})"
                elif total_score <= -3: msg = f"❄️ 종합: 폭락 주의 (패닉) ({total_score})"
                elif total_score < 0: msg = f"😨 종합: 부정적 흐름 ({total_score})"
                else: msg = "😐 종합: 관망세 (중립) (0)"
                st.info(msg)
                st.dataframe(pd.DataFrame(items), column_config={"링크": st.column_config.LinkColumn("원문", display_text="(클릭)"), "제목": st.column_config.TextColumn("뉴스 제목", width="large")}, hide_index=True, use_container_width=True)
            else: st.warning("관련 뉴스를 찾지 못했습니다.")
