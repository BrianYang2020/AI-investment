import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import hmac

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_password():
    """
    登入密碼驗證

    密碼存放於 Streamlit Secrets 的 APP_PASSWORD：
    - 本機：寫在 .streamlit/secrets.toml（勿上傳 GitHub）
    - Streamlit Cloud：App settings → Secrets

    Returns:
        bool: 是否已通過驗證
    """
    # 已登入則直接通過
    if st.session_state.get("authenticated"):
        return True

    # 讀取設定的密碼；未設定時一律拒絕進入，避免網頁在沒有保護的情況下公開
    try:
        app_password = str(st.secrets["APP_PASSWORD"])
    except Exception:
        app_password = ""

    st.title("🔒 AI 股票趨勢分析系統")

    if not app_password:
        st.error("尚未設定登入密碼。請在 .streamlit/secrets.toml（本機）或 Streamlit Cloud 的 Secrets 中加入：APP_PASSWORD = \"您的密碼\"")
        return False

    # 使用表單，按 Enter 即可送出
    with st.form("login_form"):
        password = st.text_input("請輸入登入密碼", type="password")
        submitted = st.form_submit_button("登入", type="primary")

    if submitted:
        # 使用 hmac.compare_digest 比對，避免時間差攻擊
        if hmac.compare_digest(password.encode("utf-8"), app_password.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密碼錯誤，請重新輸入。")

    return False

# 未通過密碼驗證則停止執行後續內容
if not check_password():
    st.stop()

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()

# RSI 指標預設參數
DEFAULT_RSI_PERIOD = 14       # RSI 計算天數
DEFAULT_RSI_OVERBOUGHT = 70   # 超買門檻
DEFAULT_RSI_OVERSOLD = 30     # 超賣門檻

def get_stock_data(symbol, api_key, start_date, end_date):
    """
    從Financial Modeling Prep API獲取股票歷史數據
    
    Args:
        symbol: 股票代碼
        api_key: FMP API金鑰
        start_date: 起始日期
        end_date: 結束日期
    
    Returns:
        DataFrame: 包含股票歷史數據的DataFrame
    """
    try:
        # 構建API請求URL
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            'symbol': symbol,
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        
        # 發送API請求
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # 檢查API響應 - 新API直接回傳陣列
        if not isinstance(data, list) or len(data) == 0:
            st.error(f"無法獲取股票 {symbol} 的數據，請檢查股票代碼是否正確。")
            return None
        
        # 轉換為DataFrame - 新API直接回傳歷史數據陣列
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"API請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"數據處理錯誤：{str(e)}")
        return None

def filter_by_date_range(df, start_date, end_date):
    """
    根據日期範圍過濾數據
    
    Args:
        df: 股票數據DataFrame
        start_date: 起始日期
        end_date: 結束日期
    
    Returns:
        DataFrame: 過濾後的數據
    """
    if df is None:
        return None
    
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask].copy()
    
    return filtered_df.reset_index(drop=True)

def get_moving_averages(df):
    """
    計算移動平均線（MA5, MA10, MA20, MA60）
    
    Args:
        df: 股票數據DataFrame
    
    Returns:
        DataFrame: 包含移動平均線的數據
    """
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    
    # 計算移動平均線
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()

    return df

def calculate_rsi(df, period=DEFAULT_RSI_PERIOD):
    """
    計算RSI相對強弱指標

    公式：RSI = 100 - (100 / (1 + RS))
          RS  = N日平均漲幅 / N日平均跌幅（N 預設為 14）

    Args:
        df: 股票數據DataFrame（需包含 close 欄位）
        period: RSI計算天數，預設14天

    Returns:
        DataFrame: 新增 RSI 欄位的數據；前 period 筆因資料不足為 NaN
    """
    if df is None or len(df) == 0:
        return None

    # 參數檢查：計算天數至少需要 2 天
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError(f"RSI計算天數必須為大於等於2的整數，目前為：{period}")

    if 'close' not in df.columns:
        raise ValueError("數據缺少收盤價（close）欄位，無法計算RSI")

    df = df.copy()

    # 每日收盤價變動，拆分為漲幅與跌幅（跌幅取正值）
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # N日平均漲幅與平均跌幅（資料不足 N 日時不計算）
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # 計算 RS 與 RSI，平均跌幅為 0 時避免除以零
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 特殊情況處理：
    # - 期間內無任何跌幅 → RSI = 100
    # - 期間內價格完全沒有變動 → RSI = 50（中性）
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    # 資料不足的前段維持 NaN
    rsi = rsi.where(avg_gain.notna())

    df['RSI'] = rsi

    return df

def get_rsi_status(rsi_value, overbought=DEFAULT_RSI_OVERBOUGHT, oversold=DEFAULT_RSI_OVERSOLD):
    """
    依據RSI數值判斷超買超賣狀態

    Args:
        rsi_value: RSI數值
        overbought: 超買門檻（預設70）
        oversold: 超賣門檻（預設30）

    Returns:
        str: 狀態說明（超買區 / 超賣區 / 中性區 / 資料不足）
    """
    if rsi_value is None or pd.isna(rsi_value):
        return "資料不足"
    if rsi_value > overbought:
        return "超買區"
    if rsi_value < oversold:
        return "超賣區"
    return "中性區"

def create_candlestick_chart(df, symbol, rsi_period=DEFAULT_RSI_PERIOD,
                             overbought=DEFAULT_RSI_OVERBOUGHT, oversold=DEFAULT_RSI_OVERSOLD):
    """
    創建K線圖、移動平均線、成交量與RSI圖表

    Args:
        df: 包含股票數據、移動平均線與RSI的DataFrame
        symbol: 股票代碼
        rsi_period: RSI計算天數（用於標題顯示）
        overbought: RSI超買門檻
        oversold: RSI超賣門檻

    Returns:
        plotly.graph_objects.Figure: 互動式圖表
    """
    # 圖表分為上下區塊：上方K線與成交量，下方RSI走勢圖
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=('價格與移動平均線', '成交量', f'RSI 相對強弱指標（{rsi_period}日）'),
        row_heights=[0.55, 0.15, 0.30]
    )
    
    # K線圖
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ),
        row=1, col=1
    )
    
    # 移動平均線
    ma_colors = {
        'MA5': '#ff6b6b',
        'MA10': '#4ecdc4', 
        'MA20': '#45b7d1',
        'MA60': '#96ceb4'
    }
    
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[ma],
                mode='lines',
                name=ma,
                line=dict(color=ma_colors[ma], width=2)
            ),
            row=1, col=1
        )
    
    # 成交量
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color='#a55eea',
            opacity=0.6
        ),
        row=2, col=1
    )

    # RSI 線圖（藍色），滑鼠懸停顯示 RSI 數值
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['RSI'],
                mode='lines',
                name=f'RSI({rsi_period})',
                line=dict(color='blue', width=2),
                hovertemplate='日期：%{x|%Y-%m-%d}<br>RSI：%{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )

        # 標記超買 / 超賣的交易日，作為視覺警告
        overbought_points = df[df['RSI'] > overbought]
        oversold_points = df[df['RSI'] < oversold]
        if len(overbought_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=overbought_points['date'],
                    y=overbought_points['RSI'],
                    mode='markers',
                    name='超買警告',
                    marker=dict(color='red', size=6, symbol='triangle-up'),
                    hovertemplate='日期：%{x|%Y-%m-%d}<br>RSI：%{y:.2f}（超買）<extra></extra>'
                ),
                row=3, col=1
            )
        if len(oversold_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=oversold_points['date'],
                    y=oversold_points['RSI'],
                    mode='markers',
                    name='超賣警告',
                    marker=dict(color='green', size=6, symbol='triangle-down'),
                    hovertemplate='日期：%{x|%Y-%m-%d}<br>RSI：%{y:.2f}（超賣）<extra></extra>'
                ),
                row=3, col=1
            )

    # RSI 區塊：超買區紅色背景、超賣區綠色背景
    # （需在加入RSI線之後才繪製，否則Plotly會略過空白子圖）
    fig.add_hrect(
        y0=overbought, y1=100,
        fillcolor='red', opacity=0.1, line_width=0,
        row=3, col=1
    )
    fig.add_hrect(
        y0=0, y1=oversold,
        fillcolor='green', opacity=0.1, line_width=0,
        row=3, col=1
    )

    # 超買 / 超賣門檻：紅色虛線
    fig.add_hline(
        y=overbought, line_dash='dash', line_color='red',
        annotation_text=f'超買 {overbought}', annotation_position='top left',
        row=3, col=1
    )
    fig.add_hline(
        y=oversold, line_dash='dash', line_color='red',
        annotation_text=f'超賣 {oversold}', annotation_position='bottom left',
        row=3, col=1
    )

    # 圖表標題顯示股票代碼與分析期間
    period_text = ''
    if len(df) > 0:
        period_text = f"（{df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}）"

    # 更新佈局
    fig.update_layout(
        title=f'{symbol} 股價技術分析圖表{period_text}',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # 更新x軸
    fig.update_xaxes(
        rangeslider_visible=False,
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="日期", row=3, col=1)

    # 更新y軸（RSI 固定 0~100）
    fig.update_yaxes(title_text="價格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig

def generate_ai_insights(symbol, stock_data, openai_api_key, start_date, end_date,
                         rsi_period=DEFAULT_RSI_PERIOD,
                         overbought=DEFAULT_RSI_OVERBOUGHT, oversold=DEFAULT_RSI_OVERSOLD):
    """
    使用OpenAI進行技術分析（含RSI分析）

    Args:
        symbol: 股票代碼
        stock_data: 股票數據DataFrame（含移動平均線與RSI）
        openai_api_key: OpenAI API金鑰
        start_date: 起始日期
        end_date: 結束日期
        rsi_period: RSI計算天數
        overbought: RSI超買門檻
        oversold: RSI超賣門檻

    Returns:
        str: AI分析結果
    """
    try:
        # 創建OpenAI客戶端
        client = OpenAI(api_key=openai_api_key)

        # 準備數據
        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100

        # 準備RSI摘要資訊
        rsi_series = stock_data['RSI'].dropna() if 'RSI' in stock_data.columns else pd.Series(dtype=float)
        if len(rsi_series) > 0:
            latest_rsi = rsi_series.iloc[-1]
            rsi_summary = f"""- RSI 參數：{rsi_period} 日，超買門檻 {overbought}，超賣門檻 {oversold}
- 當前RSI值：{latest_rsi:.2f}（狀態：{get_rsi_status(latest_rsi, overbought, oversold)}）
- 期間RSI最高值：{rsi_series.max():.2f}，最低值：{rsi_series.min():.2f}，平均值：{rsi_series.mean():.2f}
- 期間處於超買區（RSI>{overbought}）天數：{int((rsi_series > overbought).sum())} 天
- 期間處於超賣區（RSI<{oversold}）天數：{int((rsi_series < oversold).sum())} 天"""
        else:
            rsi_summary = f"- RSI 參數：{rsi_period} 日（期間資料不足，無法計算RSI）"

        # 轉換數據為JSON格式
        data_json = stock_data.to_json(orient='records', date_format='iso')
        
        # 構建AI提示語
        system_message = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""
        
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% (從 ${start_price:.2f} 變化到 ${end_price:.2f})

### RSI 指標摘要
{rsi_summary}

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量、移動平均線和RSI：
{data_json}

### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 4. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 5. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）
- 關鍵價位觀察點
- 技術面風險因子

#### 6. RSI分析
- 當前RSI值與狀態判斷（超買區 / 超賣區 / 中性區）
- RSI指標解讀：期間RSI走勢與歷史區間分布
- 超買超賣狀態解讀：歷史上RSI進入超買（>{overbought}）或超賣（<{oversold}）區時，後續價格曾出現的現象
- 動量分析：RSI走勢反映的價格動能強弱變化，以及RSI與價格之間是否出現背離現象

### 綜合評估要求
#### 輸出格式要求
- 必須包含獨立的「RSI分析」章節，且章節開頭明確列出當前RSI值與狀態判斷
- 條理清晰，分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測，強調分析的局限性
- 在適當位置使用表格或重點標記

分析目標：{symbol}"""
        
        # 調用OpenAI API (新版本)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI分析失敗：{str(e)}")
        return "AI分析暫時無法使用，請檢查API金鑰或稍後再試。"

# 側邊欄設置
st.sidebar.markdown("## 🔧 分析設定")

# 登出按鈕
if st.sidebar.button("🚪 登出"):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.divider()

# 輸入控制項
symbol = st.sidebar.text_input(
    "股票代碼",
    value="AAPL",
    help="輸入美股股票代碼，例如：AAPL, MSFT, GOOGL, TSLA"
)

fmp_api_key = st.sidebar.text_input(
    "FMP API Key",
    type="password",
    help="請輸入您的Financial Modeling Prep API金鑰"
)

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password",
    help="請輸入您的OpenAI API金鑰"
)

# 日期選擇
default_start_date = datetime.now() - timedelta(days=90)
default_end_date = datetime.now()

start_date = st.sidebar.date_input(
    "起始日期",
    value=default_start_date,
    help="選擇分析的起始日期"
)

end_date = st.sidebar.date_input(
    "結束日期", 
    value=default_end_date,
    help="選擇分析的結束日期"
)

# RSI 參數設定（支援自訂）
st.sidebar.markdown("#### 📉 RSI 參數設定")
rsi_period = st.sidebar.number_input(
    "RSI 計算天數",
    min_value=2,
    max_value=60,
    value=DEFAULT_RSI_PERIOD,
    step=1,
    help="RSI 計算所使用的天數，預設為 14 日"
)

rsi_overbought = st.sidebar.number_input(
    "超買門檻",
    min_value=50,
    max_value=95,
    value=DEFAULT_RSI_OVERBOUGHT,
    step=1,
    help="RSI 高於此值視為超買，預設為 70"
)

rsi_oversold = st.sidebar.number_input(
    "超賣門檻",
    min_value=5,
    max_value=50,
    value=DEFAULT_RSI_OVERSOLD,
    step=1,
    help="RSI 低於此值視為超賣，預設為 30"
)

# 分析按鈕
analyze_button = st.sidebar.button("🚀 開始分析", type="primary")

# 免責聲明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")

# 主要分析邏輯
if analyze_button:
    # 輸入驗證
    if not symbol.strip():
        st.error("請輸入股票代碼")
    elif not fmp_api_key.strip():
        st.error("請輸入FMP API Key")
    elif not openai_api_key.strip():
        st.error("請輸入OpenAI API Key")
    elif start_date >= end_date:
        st.error("起始日期不能晚於或等於結束日期")
    elif rsi_oversold >= rsi_overbought:
        st.error("RSI 超賣門檻必須小於超買門檻")
    else:
        # 開始分析流程
        with st.spinner("正在獲取股票數據..."):
            # 往前多抓一段暖身資料，讓MA60與RSI在分析起始日即有完整數值
            warmup_days = int(max(60, rsi_period) * 1.6) + 10
            fetch_start_date = start_date - timedelta(days=warmup_days)

            # 獲取股票數據
            stock_data = get_stock_data(symbol.upper(), fmp_api_key, fetch_start_date, end_date)

            if stock_data is not None and len(stock_data) > 0:
                # 先以完整數據計算技術指標，再過濾至使用者選擇的日期範圍
                data_with_ma = None
                with st.spinner("正在計算技術指標..."):
                    try:
                        full_data = get_moving_averages(stock_data)
                        full_data = calculate_rsi(full_data, int(rsi_period))
                        data_with_ma = filter_by_date_range(full_data, start_date, end_date)
                    except Exception as e:
                        st.error(f"技術指標計算失敗：{str(e)}")

                if data_with_ma is not None and len(data_with_ma) > 0:
                    st.success(f"成功獲取 {len(data_with_ma)} 筆交易數據")

                    if data_with_ma['RSI'].isna().all():
                        st.warning(f"可用交易數據不足 {int(rsi_period) + 1} 筆，無法計算 RSI，請拉長日期範圍或縮短 RSI 計算天數。")

                    if data_with_ma is not None:
                        # 顯示K線圖
                        st.markdown("### 📊 股價K線圖與技術指標")
                        chart = create_candlestick_chart(
                            data_with_ma,
                            symbol.upper(),
                            rsi_period=int(rsi_period),
                            overbought=rsi_overbought,
                            oversold=rsi_oversold
                        )
                        st.plotly_chart(chart, use_container_width=True)

                        # RSI 超買超賣狀態
                        st.markdown("### 📉 RSI 相對強弱指標")
                        rsi_valid = data_with_ma['RSI'].dropna()
                        if len(rsi_valid) > 0:
                            latest_rsi = rsi_valid.iloc[-1]
                            rsi_status = get_rsi_status(latest_rsi, rsi_overbought, rsi_oversold)

                            rsi_col1, rsi_col2, rsi_col3 = st.columns(3)
                            with rsi_col1:
                                st.metric(
                                    f"當前 RSI（{int(rsi_period)}日）",
                                    f"{latest_rsi:.2f}",
                                    help="分析期間最後一個交易日的 RSI 數值"
                                )
                            with rsi_col2:
                                st.metric(
                                    "超買天數",
                                    f"{int((rsi_valid > rsi_overbought).sum())} 天",
                                    help=f"期間內 RSI 高於 {rsi_overbought} 的交易日數"
                                )
                            with rsi_col3:
                                st.metric(
                                    "超賣天數",
                                    f"{int((rsi_valid < rsi_oversold).sum())} 天",
                                    help=f"期間內 RSI 低於 {rsi_oversold} 的交易日數"
                                )

                            # 超買 / 超賣警告
                            if rsi_status == "超買區":
                                st.warning(f"⚠️ 超買警告：當前 RSI 為 {latest_rsi:.2f}，高於超買門檻 {rsi_overbought}。技術指標反映價格短期漲幅較大。")
                            elif rsi_status == "超賣區":
                                st.warning(f"⚠️ 超賣警告：當前 RSI 為 {latest_rsi:.2f}，低於超賣門檻 {rsi_oversold}。技術指標反映價格短期跌幅較大。")
                            else:
                                st.info(f"當前 RSI 為 {latest_rsi:.2f}，位於中性區（{rsi_oversold} ~ {rsi_overbought}）。")
                        else:
                            st.info("RSI 資料不足，暫無法判斷超買超賣狀態。")
                        
                        # 基本統計資訊
                        st.markdown("### 📈 基本統計資訊")
                        col1, col2, col3 = st.columns(3)
                        
                        start_price = data_with_ma['close'].iloc[0]
                        end_price = data_with_ma['close'].iloc[-1]
                        price_change = end_price - start_price
                        price_change_pct = (price_change / start_price) * 100
                        
                        with col1:
                            st.metric(
                                "起始價格",
                                f"${start_price:.2f}",
                                help="分析期間第一個交易日的收盤價"
                            )
                        
                        with col2:
                            st.metric(
                                "結束價格",
                                f"${end_price:.2f}",
                                help="分析期間最後一個交易日的收盤價"
                            )
                        
                        with col3:
                            st.metric(
                                "價格變化",
                                f"${price_change:.2f}",
                                f"{price_change_pct:.2f}%",
                                help="期間內的價格變化金額和百分比"
                            )
                        
                        # AI技術分析
                        st.markdown("### 🤖 AI技術分析")
                        with st.spinner("AI 正在分析中..."):
                            ai_analysis = generate_ai_insights(
                                symbol.upper(), 
                                data_with_ma, 
                                openai_api_key,
                                start_date,
                                end_date,
                                rsi_period=int(rsi_period),
                                overbought=rsi_overbought,
                                oversold=rsi_oversold
                            )
                        
                        if ai_analysis:
                            st.markdown(ai_analysis)
                        
                        # 歷史數據表格
                        st.markdown("### 📋 歷史數據表格")
                        # 顯示最近10筆數據
                        display_data = data_with_ma.tail(10).copy()
                        display_data = display_data.sort_values('date', ascending=False)
                        
                        # 格式化數據
                        display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60', 'RSI']
                        display_data_formatted = display_data[display_columns].copy()

                        # 重命名欄位
                        display_data_formatted.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60', f'RSI({int(rsi_period)})']
                        
                        st.dataframe(
                            display_data_formatted,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.success("✅ 分析完成！")
                        
                elif data_with_ma is not None:
                    st.warning("所選日期範圍內沒有交易數據，請調整日期範圍。")
            else:
                st.error("無法獲取股票數據，請檢查股票代碼和API金鑰。")

# 初始頁面說明
if not analyze_button:
    st.markdown("""
    ## 歡迎使用 AI 股票趨勢分析系統 👋
    
    ### 🚀 功能特色
    - **專業K線圖表**: 互動式價格圖表，包含移動平均線技術指標
    - **AI智能分析**: 使用先進AI模型進行深度技術面分析
    - **歷史數據**: 詳細的股票歷史價格和成交量數據
    - **教育導向**: 客觀的技術分析，僅供學習研究使用
    
    ### 📝 使用方法
    1. 在左側輸入股票代碼（如：AAPL, MSFT, GOOGL）
    2. 輸入您的API金鑰（FMP和OpenAI）
    3. 選擇分析的日期範圍
    4. 點擊「開始分析」按鈕
    
    ### 💡 技術指標說明
    - **MA5**: 5日移動平均線，短期趨勢指標
    - **MA10**: 10日移動平均線，短中期趨勢指標  
    - **MA20**: 20日移動平均線，中期趨勢指標
    - **MA60**: 60日移動平均線，長期趨勢指標
    - **RSI**: 相對強弱指標（預設14日），RSI = 100 - 100 / (1 + RS)，RS = 平均漲幅 / 平均跌幅；高於70為超買區、低於30為超賣區（門檻可於側邊欄自訂）

    ### 🔑 API金鑰獲取
    - **FMP API**: 前往 [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs) 註冊
    - **OpenAI API**: 前往 [OpenAI Platform](https://platform.openai.com) 註冊
    
    ---
    **開始您的技術分析之旅吧！** 📈
    """)