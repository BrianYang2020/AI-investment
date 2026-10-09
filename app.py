import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()

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

def create_candlestick_chart(df, symbol):
    """
    創建K線圖和移動平均線圖表
    
    Args:
        df: 包含股票數據和移動平均線的DataFrame
        symbol: 股票代碼
    
    Returns:
        plotly.graph_objects.Figure: 互動式圖表
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('價格與移動平均線', '成交量'),
        row_width=[0.2, 0.7]
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
    
    # 更新佈局
    fig.update_layout(
        title=f'{symbol} 股價技術分析圖表',
        xaxis_title='日期',
        yaxis_title='價格 (USD)',
        height=700,
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
    
    # 更新y軸
    fig.update_yaxes(title_text="價格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig

def generate_ai_insights(symbol, stock_data, openai_api_key, start_date, end_date):
    """
    使用OpenAI進行技術分析
    
    Args:
        symbol: 股票代碼
        stock_data: 股票數據DataFrame
        openai_api_key: OpenAI API金鑰
        start_date: 起始日期
        end_date: 結束日期
    
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

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量和移動平均線：
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

### 綜合評估要求
#### 輸出格式要求
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
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI分析失敗：{str(e)}")
        return "AI分析暫時無法使用，請檢查API金鑰或稍後再試。"

# 側邊欄設置
st.sidebar.markdown("## 🔧 分析設定")
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
    else:
        # 開始分析流程
        with st.spinner("正在獲取股票數據..."):
            # 獲取股票數據
            stock_data = get_stock_data(symbol.upper(), fmp_api_key, start_date, end_date)
            
            if stock_data is not None and len(stock_data) > 0:
                st.success(f"成功獲取 {len(stock_data)} 筆交易數據")
                
                # 過濾數據
                filtered_data = filter_by_date_range(stock_data, start_date, end_date)
                
                if filtered_data is not None and len(filtered_data) > 0:
                    # 計算移動平均線
                    with st.spinner("正在計算技術指標..."):
                        data_with_ma = get_moving_averages(filtered_data)
                    
                    if data_with_ma is not None:
                        # 顯示K線圖
                        st.markdown("### 📊 股價K線圖與技術指標")
                        chart = create_candlestick_chart(data_with_ma, symbol.upper())
                        st.plotly_chart(chart, use_container_width=True)
                        
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
                                end_date
                            )
                        
                        if ai_analysis:
                            st.markdown(ai_analysis)
                        
                        # 歷史數據表格
                        st.markdown("### 📋 歷史數據表格")
                        # 顯示最近10筆數據
                        display_data = data_with_ma.tail(10).copy()
                        display_data = display_data.sort_values('date', ascending=False)
                        
                        # 格式化數據
                        display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60']
                        display_data_formatted = display_data[display_columns].copy()
                        
                        # 重命名欄位
                        display_data_formatted.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60']
                        
                        st.dataframe(
                            display_data_formatted,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.success("✅ 分析完成！")
                        
                else:
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
    
    ### 🔑 API金鑰獲取
    - **FMP API**: 前往 [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs) 註冊
    - **OpenAI API**: 前往 [OpenAI Platform](https://platform.openai.com) 註冊
    
    ---
    **開始您的技術分析之旅吧！** 📈
    """)