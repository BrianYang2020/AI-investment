import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import traceback
import requests
from openai import OpenAI
from datetime import datetime, timedelta
import numpy as np
import re
import hmac

# 設置頁面配置
st.set_page_config(page_title="AI 分析台股基本面應用", layout="wide")

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

    st.header("🔒 【Code Gym】AI 分析台股基本面應用", divider="rainbow")

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

st.header("【Code Gym】AI 分析台股基本面應用", divider="rainbow")

# =====================================================================
# 常數設定
# =====================================================================

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
API_TIMEOUT_SECONDS = 30            # API 連線逾時秒數，避免網頁卡住
AI_MODEL_OPTIONS = ["o4-mini", "gpt-4o-mini"]
FINANCIAL_INDUSTRY_KEYWORDS = ["金融", "保險", "銀行", "證券"]
PAR_VALUE_PER_SHARE = 10            # 台股普通股面額（多數為 10 元），用於由股本推算股數

# FinMind type 欄位 → 內部標準欄位（依報表分開對應）
# 注意：EquityAttributableToOwnersOfParent 在損益表是「淨利歸屬於母公司業主」，
#       在資產負債表是「歸屬於母公司業主之權益」，必須分開處理，否則會互相覆蓋。
INCOME_FIELD_MAPPING = {
    'Revenue': 'revenues',                                  # 營業收入
    'GrossProfit': 'grossprofit',                           # 營業毛利
    'OperatingIncome': 'operatingincomeloss',               # 營業利益
    'PreTaxIncome': 'pretax_income',                        # 稅前淨利
    'IncomeAfterTaxes': 'netincomeloss',                    # 本期淨利（含非控制權益）
    'EquityAttributableToOwnersOfParent': 'netincome_parent',  # 淨利歸屬於母公司業主
    'EPS': 'eps',                                           # 基本每股盈餘
}

BALANCE_FIELD_MAPPING = {
    'TotalAssets': 'assets',                                # 資產總額
    'Liabilities': 'liabilities',                           # 負債總額
    'Equity': 'stockholdersequity',                         # 權益總額
    'EquityAttributableToOwnersOfParent': 'equity_parent',  # 歸屬於母公司業主之權益
    'CurrentAssets': 'assetscurrent',                       # 流動資產
    'CurrentLiabilities': 'liabilitiescurrent',             # 流動負債
    'RetainedEarnings': 'retainedearningsaccumulateddeficit',  # 保留盈餘
    'BondsPayable': 'bonds_payable',                        # 應付公司債
    'LongtermBorrowings': 'longterm_borrowings',            # 長期借款
    'CapitalStock': 'capital_stock',                        # 股本合計
    'OrdinaryShare': 'ordinary_share',                      # 普通股股本
}

CASHFLOW_FIELD_MAPPING = {
    'CashFlowsFromOperatingActivities': 'netcashprovidedbyusedinoperatingactivities',  # 營業活動現金流
    'CashProvidedByInvestingActivities': 'netcashprovidedbyusedininvestingactivities',  # 投資活動現金流
    'CashFlowsProvidedFromFinancingActivities': 'netcashprovidedbyusedinfinancingactivities',  # 籌資活動現金流
    'PropertyAndPlantAndEquipment': 'capex_raw',            # 取得不動產、廠房及設備（負值）
    'InterestExpense': 'interest_expense',                  # 利息費用
    'PayTheInterest': 'interest_paid_raw',                  # 支付之利息（負值，利息費用缺漏時備用）
    'CashBalancesBeginningOfPeriod': 'cash_beginning',      # 期初現金（用於判斷是否為年初累計）
}

# 流量欄位（損益表、現金流量表）：分析時以「近四季合計（TTM）」計算
FLOW_FIELDS = [
    'revenues', 'grossprofit', 'operatingincomeloss', 'pretax_income',
    'netincomeloss', 'netincome_parent', 'eps',
    'netcashprovidedbyusedinoperatingactivities',
    'netcashprovidedbyusedininvestingactivities',
    'netcashprovidedbyusedinfinancingactivities',
    'capex', 'interest_expense',
]

# 存量欄位（資產負債表）：分析時取期末數
STOCK_FIELDS = [
    'assets', 'liabilities', 'stockholdersequity', 'equity_parent',
    'assetscurrent', 'liabilitiescurrent', 'retainedearningsaccumulateddeficit',
    'longterm_debt', 'share_capital',
]

# =====================================================================
# 工具函數
# =====================================================================

def is_missing(value):
    """判斷數值是否缺漏（None 或 NaN）"""
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))

def format_large_number(num):
    """
    將大數字轉換為易讀格式（新台幣元）
    1 兆 = 1e12、1 億 = 1e8、1 萬 = 1e4
    """
    if is_missing(num):
        return "N/A"
    sign = "-" if num < 0 else ""
    value = abs(num)
    if value >= 1e12:
        return f"{sign}{value / 1e12:,.2f}兆"
    if value >= 1e8:
        return f"{sign}{value / 1e8:,.2f}億"
    if value >= 1e4:
        return f"{sign}{value / 1e4:,.2f}萬"
    return f"{sign}{value:,.0f}"

def format_ratio(value, digits=4):
    """格式化比率，缺漏時顯示 N/A"""
    return "N/A" if is_missing(value) else f"{value:.{digits}f}"

def format_percent(value):
    """格式化百分比，缺漏時顯示 N/A"""
    return "N/A" if is_missing(value) else f"{value:.2%}"

def safe_divide(numerator, denominator):
    """安全除法：任一值缺漏或分母為 0 時回傳 None（不以 0 代替，避免誤判）"""
    if is_missing(numerator) or is_missing(denominator) or denominator == 0:
        return None
    return numerator / denominator

def compare_greater(a, b):
    """
    比較 a > b，任一值缺漏時回傳 None（表示無法判斷）
    差異小於相對誤差 1e-9 視為相等，避免浮點數誤差被判定為「改善」
    """
    if is_missing(a) or is_missing(b):
        return None
    tolerance = 1e-9 * max(abs(a), abs(b), 1e-12)
    return a - b > tolerance

def validate_taiwan_stock_code(stock_code):
    """驗證台股代碼是否為四位數字格式"""
    if not stock_code:
        return False, "請輸入股票代碼"

    stock_code = stock_code.strip()

    if not re.match(r'^\d{4}$', stock_code):
        return False, "台股代碼必須為四位數字格式（例如：2330、2454、2317、2412）"

    return True, ""

def period_label(period):
    """季度顯示標籤，例如 2025Q1"""
    return f"{period.year}Q{period.quarter}"

# =====================================================================
# FinMind API 整合模組
# =====================================================================

def fetch_finmind_dataset(dataset, stock_id, api_token, start_date, end_date=None):
    """
    呼叫 FinMind 統一端點取得單一 dataset

    Returns:
        list: API 回傳的 data 陣列
    """
    params = {
        'dataset': dataset,
        'data_id': stock_id,
        'start_date': start_date,
        'token': api_token,
    }
    if end_date:
        params['end_date'] = end_date

    try:
        response = requests.get(FINMIND_API_URL, params=params, timeout=API_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        raise Exception(f"{dataset} 連線逾時（超過 {API_TIMEOUT_SECONDS} 秒），請稍後再試")
    except requests.exceptions.RequestException as e:
        raise Exception(f"{dataset} 連線失敗：{e}")

    # 嘗試讀取 FinMind 回傳的錯誤訊息
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code != 200 or payload.get('status') != 200:
        message = payload.get('msg', f"HTTP {response.status_code}")
        hint = ""
        if response.status_code == 402 or 'limit' in str(message).lower():
            hint = "（已超過 FinMind 使用次數上限，請稍後再試或升級方案）"
        elif response.status_code in (400, 401, 403):
            hint = "（請確認 FinMind API Token 是否正確）"
        raise Exception(f"{dataset} API 回傳錯誤：{message}{hint}")

    return payload.get('data', [])

@st.cache_data(ttl=3600, show_spinner=False)
def get_finmind_raw_data(stock_id, api_token, start_date):
    """
    從 FinMind 取得完整原始資料（快取 1 小時，同一檔股票重複查詢不再呼叫 API）
    """
    datasets = {
        'financial_statements': 'TaiwanStockFinancialStatements',
        'balance_sheet': 'TaiwanStockBalanceSheet',
        'cash_flow': 'TaiwanStockCashFlowsStatement',
        'stock_info': 'TaiwanStockInfo',
        'key_metrics': 'TaiwanStockPER',
    }

    raw_data = {}
    for data_type, dataset in datasets.items():
        raw_data[data_type] = fetch_finmind_dataset(dataset, stock_id, api_token, start_date)

    # 近 30 天股價（用於顯示最新收盤價；失敗不影響主要分析）
    try:
        price_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        raw_data['stock_price'] = fetch_finmind_dataset('TaiwanStockPrice', stock_id, api_token, price_start)
    except Exception:
        raw_data['stock_price'] = []

    return raw_data

# =====================================================================
# 資料處理模組：欄位對應、單季化、TTM
# =====================================================================

def build_statement_frame(records, field_mapping):
    """
    將 FinMind 長格式資料（date / type / value）轉為以季度為索引的寬表
    """
    rows = [
        {'date': item['date'], 'field': field_mapping[item['type']], 'value': item.get('value')}
        for item in records
        if item.get('type') in field_mapping
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['period'] = pd.to_datetime(df['date']).dt.to_period('Q')
    wide = df.pivot_table(index='period', columns='field', values='value', aggfunc='last')
    return wide.sort_index()

def detect_cumulative_by_ratio(series):
    """
    以「同年度後期數值 ÷ 第一季數值」判斷是否為年初累計值
    - 累計值：第 n 季約為第一季的 n 倍
    - 單季值：各季數值相近

    Returns:
        True（累計）/ False（單季）/ None（資料不足無法判斷）
    """
    series = series.dropna()
    cumulative_votes = single_votes = 0
    for _, group in series.groupby(series.index.year):
        values = {period.quarter: value for period, value in group.items()}
        if values.get(1, 0) <= 0:
            continue
        later_quarters = [q for q, v in values.items() if q >= 2 and v > 0]
        if not later_quarters:
            continue
        n = max(later_quarters)
        ratio = values[n] / values[1]
        if ratio > (1 + n) / 2:
            cumulative_votes += 1
        else:
            single_votes += 1
    if cumulative_votes == 0 and single_votes == 0:
        return None
    return cumulative_votes > single_votes

def detect_cashflow_cumulative(cashflow_df):
    """
    判斷現金流量表是否為年初累計值
    1. 優先使用「期初現金」：累計報表同一年度各季的期初現金皆為年初數，會相同
    2. 其次以營業現金流比例判斷
    3. 仍無法判斷時，依台灣財報慣例假設為累計值
    """
    if 'cash_beginning' in cashflow_df:
        beginning = cashflow_df['cash_beginning'].dropna()
        same_years = different_years = 0
        for _, group in beginning.groupby(beginning.index.year):
            if len(group) >= 2:
                if np.allclose(group.values, group.values[0], rtol=1e-6):
                    same_years += 1
                else:
                    different_years += 1
        if same_years or different_years:
            return same_years > different_years

    operating_field = 'netcashprovidedbyusedinoperatingactivities'
    if operating_field in cashflow_df:
        result = detect_cumulative_by_ratio(cashflow_df[operating_field])
        if result is not None:
            return result

    return True

def decumulate(df, fields):
    """
    將年初累計值轉換為單季值：Q1 = 累計Q1；Qn = 累計Qn − 累計Q(n−1)
    前一季缺漏時該季設為 NaN（無法還原）
    """
    full_index = pd.period_range(df.index.min(), df.index.max(), freq='Q')
    full = df.reindex(full_index)
    is_first_quarter = full.index.quarter == 1
    for field in fields:
        if field not in full:
            continue
        cumulative = full[field]
        single = cumulative - cumulative.shift(1)
        full[field] = single.where(~is_first_quarter, cumulative)
    return full.loc[df.index]

def build_quarterly_data(raw_data):
    """
    合併三大報表為單季資料表，並建立衍生欄位

    Returns:
        (quarterly_df, conversion_notes)
    """
    income = build_statement_frame(raw_data.get('financial_statements', []), INCOME_FIELD_MAPPING)
    balance = build_statement_frame(raw_data.get('balance_sheet', []), BALANCE_FIELD_MAPPING)
    cashflow = build_statement_frame(raw_data.get('cash_flow', []), CASHFLOW_FIELD_MAPPING)

    if income.empty or 'revenues' not in income:
        raise Exception("無法獲取損益表數據，請檢查股票代碼是否正確")

    notes = []

    # 損益表：判斷是否為年初累計值
    if detect_cumulative_by_ratio(income['revenues']):
        income = decumulate(income, income.columns)
        notes.append("損益表偵測為年初累計值，已轉換為單季數值")
    else:
        notes.append("損益表為單季數值")

    # 現金流量表：台灣財報通常為年初累計值，轉換為單季
    if not cashflow.empty:
        flow_columns = [c for c in cashflow.columns if c != 'cash_beginning']
        if detect_cashflow_cumulative(cashflow):
            cashflow = decumulate(cashflow, flow_columns)
            notes.append("現金流量表偵測為年初累計值（台灣財報慣例），已轉換為單季數值")
        else:
            notes.append("現金流量表為單季數值")

    quarterly = income.join(balance, how='outer').join(cashflow, how='outer').sort_index()
    # 補齊缺漏季度，讓 TTM 計算能正確識別資料斷層
    quarterly = quarterly.reindex(pd.period_range(quarterly.index.min(), quarterly.index.max(), freq='Q'))

    def column(name):
        return quarterly[name] if name in quarterly else pd.Series(np.nan, index=quarterly.index)

    # 資本支出：取得不動產、廠房及設備（取絕對值）
    quarterly['capex'] = column('capex_raw').abs()

    # 利息費用：優先使用現金流量表的「利息費用」，缺漏時以「支付之利息」替代
    interest = column('interest_expense')
    quarterly['interest_expense'] = interest.fillna(column('interest_paid_raw').abs())

    # 長期負債 = 應付公司債 + 長期借款（FinMind 不列出金額為 0 的科目，缺漏視為 0）
    has_balance = column('assets').notna()
    long_term_debt = column('bonds_payable').fillna(0) + column('longterm_borrowings').fillna(0)
    quarterly['longterm_debt'] = long_term_debt.where(has_balance)

    # 股本：優先使用股本合計，缺漏時使用普通股股本
    quarterly['share_capital'] = column('capital_stock').fillna(column('ordinary_share'))

    for field in FLOW_FIELDS + STOCK_FIELDS:
        if field not in quarterly:
            quarterly[field] = np.nan

    return quarterly, notes

def get_pbr_asof(key_metrics, target_date, max_gap_days=10):
    """
    取得指定日期（含）之前最近一個交易日的 PBR
    財報日可能遇到週末或假日，因此不能要求日期完全相同
    """
    best_date, best_pbr = None, None
    for item in key_metrics:
        pbr = item.get('PBR')
        if not pbr:
            continue
        item_date = pd.Timestamp(item['date'])
        if item_date <= target_date and (best_date is None or item_date > best_date):
            best_date, best_pbr = item_date, pbr
    if best_date is None or (target_date - best_date).days > max_gap_days:
        return None
    return float(best_pbr)

def build_annual_views(quarterly, key_metrics, max_views=5):
    """
    建立年度分析資料：以最新一季為基準，每隔四季取一個點
    - 損益、現金流：近四季合計（TTM）
    - 資產負債：期末數

    Returns:
        list[dict]: 最新在前，[0] 為最近四季，[1] 為一年前的近四季，依此類推
    """
    ttm = quarterly[FLOW_FIELDS].rolling(4, min_periods=4).sum()

    # 基準季度：最近一個「TTM 營收、TTM 淨利、總資產」都齊全的季度
    valid = ttm['revenues'].notna() & ttm['netincomeloss'].notna() & quarterly['assets'].notna()
    if not valid.any():
        return []
    anchor = valid[valid].index.max()

    views = []
    for i in range(max_views):
        period = anchor - 4 * i
        if period < quarterly.index.min():
            break
        view = {
            'date': f"{period_label(period)}（近四季）",
            'period_end': period.end_time.normalize(),
        }
        for field in FLOW_FIELDS:
            value = ttm.at[period, field]
            view[field] = None if pd.isna(value) else float(value)
        for field in STOCK_FIELDS:
            value = quarterly.at[period, field]
            view[field] = None if pd.isna(value) else float(value)

        # 推算股數 = 股本 ÷ 面額
        view['shares_estimated'] = safe_divide(view['share_capital'], PAR_VALUE_PER_SHARE)

        # 市值 = 財報日最近交易日 PBR × 歸屬母公司權益
        pbr = get_pbr_asof(key_metrics, view['period_end'])
        equity_for_pbr = view['equity_parent'] if not is_missing(view['equity_parent']) else view['stockholdersequity']
        view['pbr'] = pbr
        view['market_capitalization'] = pbr * equity_for_pbr if pbr and not is_missing(equity_for_pbr) else None

        views.append(view)

    return views

def is_financial_industry(stock_info):
    """判斷是否為金融保險業（三大報表結構不同，F-Score / Z-Score 不適用）"""
    categories = [item.get('industry_category', '') for item in stock_info]
    return any(keyword in category for category in categories for keyword in FINANCIAL_INDUSTRY_KEYWORDS)

# =====================================================================
# 數據品質檢查
# =====================================================================

def analyze_data_quality(quarterly, annual_views, conversion_notes, is_financial):
    """分析財務數據品質並生成報告"""

    reported_quarters = quarterly['revenues'].dropna()

    quality_report = {
        "數據完整性": "良好",
        "財報季數": len(reported_quarters),
        "最新財報季度": period_label(reported_quarters.index.max()) if len(reported_quarters) else "N/A",
        "分析基準": "損益與現金流採近四季合計（TTM），資產負債採期末數，年度比較為與一年前同期比較",
        "可比較年度數": len(annual_views),
        "缺失欄位": [],
        "數據警告": [],
        "資料轉換說明": list(conversion_notes),
        "計算欄位說明": [
            "股數：以股本 ÷ 面額 10 元推算（F-Score 股份稀釋檢查直接比較股本）",
            "利息費用：取自現金流量表「利息費用」，缺漏時以「支付之利息」替代",
            "EBIT：稅前淨利 + 利息費用",
            "長期負債：應付公司債 + 長期借款（無資料視為 0）",
            "市值：財報日最近交易日之 PBR × 歸屬母公司權益，為近似值",
        ],
    }

    # 檢查最近兩個年度的關鍵欄位
    required_fields = {
        'revenues': '營收', 'netincomeloss': '淨利', 'assets': '總資產',
        'stockholdersequity': '股東權益', 'netcashprovidedbyusedinoperatingactivities': '營運現金流',
        'grossprofit': '毛利', 'assetscurrent': '流動資產', 'liabilitiescurrent': '流動負債',
        'market_capitalization': '市值',
    }
    for field, name in required_fields.items():
        missing = [view['date'] for view in annual_views[:2] if is_missing(view.get(field))]
        if missing:
            quality_report["缺失欄位"].append(f"{name}：{', '.join(missing)}")

    # 數據合理性檢查
    for view in annual_views[:2]:
        if not is_missing(view.get('assets')) and view['assets'] <= 0:
            quality_report["數據警告"].append(f"{view['date']} 總資產小於等於 0，數據可能異常")
        if not is_missing(view.get('revenues')) and view['revenues'] < 0:
            quality_report["數據警告"].append(f"{view['date']} 營收為負值，數據可能異常")

    if is_financial:
        quality_report["數據警告"].append("金融保險業的財報結構不同（無毛利、流動資產等科目），F-Score 與 Z-Score 不適用，結果僅供參考")

    # 數據年份檢查
    if len(annual_views) < 2:
        quality_report["數據警告"].append("財報少於 8 季，無法進行年對年比較分析，請將起始日期提前")
        quality_report["數據完整性"] = "嚴重不足"
    elif quality_report["缺失欄位"]:
        quality_report["數據完整性"] = "部分缺失"

    return quality_report

# =====================================================================
# 財務計算模組：四階段分析
# =====================================================================

def make_score_item(description, passed, current_value, previous_value="-"):
    """建立 F-Score 單項結果；passed 為 None 表示資料不足無法判斷"""
    item = {
        '指標': description,
        '本期': current_value,
        '前期／比較': previous_value,
    }
    if passed is None:
        item['得分'] = 0
        item['狀態'] = '－ 資料不足'
    else:
        item['得分'] = 1 if passed else 0
        item['狀態'] = '✓' if passed else '✗'
    return item

def calculate_piotroski_fscore(annual_views):
    """
    計算 Piotroski F-Score（9 項指標）
    current = 最近四季，previous = 一年前的近四季（年對年比較）
    """
    if len(annual_views) < 2:
        return None

    current = annual_views[0]
    previous = annual_views[1]

    # 獲利能力指標（4項）
    current_roa = safe_divide(current['netincomeloss'], current['assets'])
    previous_roa = safe_divide(previous['netincomeloss'], previous['assets'])
    operating_cf = current['netcashprovidedbyusedinoperatingactivities']
    net_income = current['netincomeloss']

    profitability_scores = [
        make_score_item('ROA正值檢查', compare_greater(current_roa, 0), format_ratio(current_roa)),
        make_score_item('營運現金流正值檢查', compare_greater(operating_cf, 0), format_large_number(operating_cf)),
        make_score_item('ROA年增率檢查', compare_greater(current_roa, previous_roa),
                        format_ratio(current_roa), format_ratio(previous_roa)),
        make_score_item('營運現金流品質檢查（營運現金流 > 淨利）', compare_greater(operating_cf, net_income),
                        format_large_number(operating_cf), format_large_number(net_income)),
    ]

    # 槓桿與流動性指標（3項）
    current_ltd_ratio = safe_divide(current['longterm_debt'], current['assets'])
    previous_ltd_ratio = safe_divide(previous['longterm_debt'], previous['assets'])
    current_ratio_now = safe_divide(current['assetscurrent'], current['liabilitiescurrent'])
    current_ratio_prev = safe_divide(previous['assetscurrent'], previous['liabilitiescurrent'])
    capital_now = current['share_capital']
    capital_prev = previous['share_capital']
    shares_not_increased = None if is_missing(capital_now) or is_missing(capital_prev) else capital_now <= capital_prev

    leverage_scores = [
        make_score_item('長期負債比率改善檢查', compare_greater(previous_ltd_ratio, current_ltd_ratio),
                        format_ratio(current_ltd_ratio), format_ratio(previous_ltd_ratio)),
        make_score_item('流動比率改善檢查', compare_greater(current_ratio_now, current_ratio_prev),
                        format_ratio(current_ratio_now, 2), format_ratio(current_ratio_prev, 2)),
        make_score_item('股份稀釋檢查（股本未增加）', shares_not_increased,
                        format_large_number(capital_now), format_large_number(capital_prev)),
    ]

    # 營運效率指標（2項）
    gross_margin_now = safe_divide(current['grossprofit'], current['revenues'])
    gross_margin_prev = safe_divide(previous['grossprofit'], previous['revenues'])
    turnover_now = safe_divide(current['revenues'], current['assets'])
    turnover_prev = safe_divide(previous['revenues'], previous['assets'])

    efficiency_scores = [
        make_score_item('毛利率改善檢查', compare_greater(gross_margin_now, gross_margin_prev),
                        format_ratio(gross_margin_now), format_ratio(gross_margin_prev)),
        make_score_item('資產周轉率改善檢查', compare_greater(turnover_now, turnover_prev),
                        format_ratio(turnover_now), format_ratio(turnover_prev)),
    ]

    all_items = profitability_scores + leverage_scores + efficiency_scores
    return {
        'current_date': current['date'],
        'previous_date': previous['date'],
        'profitability_scores': profitability_scores,
        'leverage_scores': leverage_scores,
        'efficiency_scores': efficiency_scores,
        'total_score': sum(item['得分'] for item in all_items),
        'evaluated_count': sum(1 for item in all_items if item['狀態'] != '－ 資料不足'),
    }

def calculate_altman_zscore(annual_views):
    """
    計算 Altman Z-Score：Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    使用近四季合計（TTM）的 EBIT 與營收，與年度門檻一致
    """
    if not annual_views:
        return None

    current = annual_views[0]
    total_assets = current['assets']

    # 營運資本 = 流動資產 − 流動負債
    working_capital = None
    if not is_missing(current['assetscurrent']) and not is_missing(current['liabilitiescurrent']):
        working_capital = current['assetscurrent'] - current['liabilitiescurrent']

    # EBIT = 稅前淨利 + 利息費用（利息費用缺漏時以 0 計算）
    ebit = None
    if not is_missing(current['pretax_income']):
        ebit = current['pretax_income'] + (current['interest_expense'] or 0)

    components_input = {
        'A': ('營運資本/總資產', working_capital, total_assets, 1.2),
        'B': ('保留盈餘/總資產', current['retainedearningsaccumulateddeficit'], total_assets, 1.4),
        'C': ('EBIT/總資產', ebit, total_assets, 3.3),
        'D': ('市值/總負債', current['market_capitalization'], current['liabilities'], 0.6),
        'E': ('營收/總資產', current['revenues'], total_assets, 1.0),
    }

    components = {}
    missing_components = []
    for key, (description, numerator, denominator, weight) in components_input.items():
        ratio = safe_divide(numerator, denominator)
        if ratio is None:
            missing_components.append(f"{key}項（{description}）")
        components[key] = {
            'ratio': ratio,
            'weighted': None if ratio is None else ratio * weight,
            'description': description,
        }

    base_data = {
        'working_capital': working_capital,
        'total_assets': total_assets,
        'retained_earnings': current['retainedearningsaccumulateddeficit'],
        'ebit': ebit,
        'market_cap': current['market_capitalization'],
        'total_liabilities': current['liabilities'],
        'revenues': current['revenues'],
    }

    # 任一組成要素缺漏則不計算總分，避免以 0 代入造成「危險區域」誤判
    if missing_components:
        return {
            'z_score': None,
            'risk_level': '無法計算',
            'risk_emoji': '❔',
            'missing_components': missing_components,
            'components': components,
            'base_data': base_data,
            'date': current['date'],
        }

    z_score = sum(component['weighted'] for component in components.values())

    if z_score > 2.99:
        risk_level, risk_emoji = "安全區域", "😊"
    elif z_score >= 1.81:
        risk_level, risk_emoji = "灰色區域", "😐"
    else:
        risk_level, risk_emoji = "危險區域", "😰"

    return {
        'z_score': z_score,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'missing_components': [],
        'components': components,
        'base_data': base_data,
        'date': current['date'],
    }

def calculate_dupont_analysis(annual_views):
    """計算杜邦分析（ROE 三因子分解），取最近 3 個年度（近四季）"""

    dupont_data = []
    for view in annual_views[:3]:
        net_margin = safe_divide(view['netincomeloss'], view['revenues'])
        asset_turnover = safe_divide(view['revenues'], view['assets'])
        equity_multiplier = safe_divide(view['assets'], view['stockholdersequity'])
        calculated_roe = None
        if None not in (net_margin, asset_turnover, equity_multiplier):
            calculated_roe = net_margin * asset_turnover * equity_multiplier

        dupont_data.append({
            'date': view['date'],
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'calculated_roe': calculated_roe,
            'direct_roe': safe_divide(view['netincomeloss'], view['stockholdersequity']),
        })

    # 趨勢變化（最新年度 − 前一年度）
    trends = []
    if len(dupont_data) >= 2:
        current, previous = dupont_data[0], dupont_data[1]
        for factor, key in [('淨利率變化', 'net_margin'), ('資產周轉率變化', 'asset_turnover'),
                            ('權益乘數變化', 'equity_multiplier'), ('ROE變化', 'direct_roe')]:
            change = None
            if not is_missing(current[key]) and not is_missing(previous[key]):
                change = current[key] - previous[key]
            trends.append({'factor': factor, 'change': change})

    return {'annual_data': dupont_data, 'trends': trends}

def calculate_cashflow_analysis(annual_views):
    """計算現金流分析（近四季合計）"""

    if not annual_views:
        return None

    current = annual_views[0]
    operating_cf = current['netcashprovidedbyusedinoperatingactivities']
    investing_cf = current['netcashprovidedbyusedininvestingactivities']
    financing_cf = current['netcashprovidedbyusedinfinancingactivities']
    net_income = current['netincomeloss']
    capex = current['capex']

    # 營運現金流品質比率 = 營運現金流 ÷ 淨利（淨利為負時比率無意義）
    cf_quality_ratio = safe_divide(operating_cf, net_income) if not is_missing(net_income) and net_income > 0 else None

    # 自由現金流 = 營運現金流 − |資本支出|
    free_cashflow = None
    if not is_missing(operating_cf):
        free_cashflow = operating_cf - abs(capex or 0)

    if cf_quality_ratio is None:
        quality_assessment, quality_emoji = "無法評估", "❔"
    elif cf_quality_ratio >= 1.2:
        quality_assessment, quality_emoji = "優秀", "😊"
    elif cf_quality_ratio >= 1.0:
        quality_assessment, quality_emoji = "良好", "🙂"
    elif cf_quality_ratio >= 0.8:
        quality_assessment, quality_emoji = "尚可", "😐"
    else:
        quality_assessment, quality_emoji = "需關注", "😰"

    total_cf = None
    if not any(is_missing(v) for v in (operating_cf, investing_cf, financing_cf)):
        total_cf = operating_cf + investing_cf + financing_cf

    return {
        'date': current['date'],
        'cf_quality_ratio': cf_quality_ratio,
        'free_cashflow': free_cashflow,
        'quality_assessment': quality_assessment,
        'quality_emoji': quality_emoji,
        'structure_analysis': [
            {'type': '營運現金流', 'amount': operating_cf},
            {'type': '投資現金流', 'amount': investing_cf},
            {'type': '融資現金流', 'amount': financing_cf},
        ],
        'detailed_data': {
            'operating_cf': operating_cf,
            'investing_cf': investing_cf,
            'financing_cf': financing_cf,
            'net_income': net_income,
            'capex': capex,
            'total_cf': total_cf,
        },
    }

# =====================================================================
# 視覺化模組
# =====================================================================

def build_display_tables(quarterly):
    """
    建立三大報表的單季展示表（單位：億元；EPS 單位：元）
    回傳日期升冪的資料（圖表用），表格展示時再反轉為降冪
    """
    data = quarterly[quarterly['revenues'].notna() | quarterly['assets'].notna()].copy()
    index = [period_label(p) for p in data.index]
    to_hundred_million = 1e8

    income_df = pd.DataFrame({
        '營收': data['revenues'] / to_hundred_million,
        '毛利': data['grossprofit'] / to_hundred_million,
        '營業利益': data['operatingincomeloss'] / to_hundred_million,
        '稅後淨利': data['netincomeloss'] / to_hundred_million,
        'EPS（元）': data['eps'],
    })
    balance_df = pd.DataFrame({
        '總資產': data['assets'] / to_hundred_million,
        '流動資產': data['assetscurrent'] / to_hundred_million,
        '總負債': data['liabilities'] / to_hundred_million,
        '流動負債': data['liabilitiescurrent'] / to_hundred_million,
        '股東權益': data['stockholdersequity'] / to_hundred_million,
    })
    ratio_df = pd.DataFrame({
        '流動比率': data['assetscurrent'] / data['liabilitiescurrent'],
        '負債比率': data['liabilities'] / data['assets'],
        '權益比率': data['stockholdersequity'] / data['assets'],
    })
    operating_cf = data['netcashprovidedbyusedinoperatingactivities']
    cash_df = pd.DataFrame({
        '營運現金流': operating_cf / to_hundred_million,
        '投資現金流': data['netcashprovidedbyusedininvestingactivities'] / to_hundred_million,
        '融資現金流': data['netcashprovidedbyusedinfinancingactivities'] / to_hundred_million,
        '資本支出': data['capex'] / to_hundred_million,
        '自由現金流': (operating_cf - data['capex'].fillna(0)) / to_hundred_million,
    })

    for df in (income_df, balance_df, ratio_df, cash_df):
        df.index = index
        df.index.name = '季度'

    return income_df, balance_df, ratio_df, cash_df

def descending(df):
    """表格展示用：最新季度在前，數值四捨五入至小數點後兩位"""
    return df.iloc[::-1].round(2)

def create_financial_charts(income_df, balance_df, cash_df):
    """創建專業財務圖表（單季數值，單位：億元）"""

    charts = {}
    common_layout = dict(xaxis_title="季度", yaxis_title="金額（億元）", template='plotly_white', height=500)

    # 損益表柱狀圖
    fig_income = go.Figure()
    income_colors = {'營收': 'steelblue', '毛利': 'darkgreen', '營業利益': 'goldenrod', '稅後淨利': 'darkred'}
    for column, color in income_colors.items():
        fig_income.add_trace(go.Bar(x=income_df.index, y=income_df[column], name=column, marker_color=color))
    fig_income.update_layout(title="損益表關鍵指標趨勢（單季）", barmode='group', **common_layout)
    charts['income'] = fig_income

    # 資產負債表趨勢圖
    fig_balance = go.Figure()
    balance_colors = {'總資產': 'steelblue', '總負債': 'darkred', '股東權益': 'darkgreen'}
    for column, color in balance_colors.items():
        fig_balance.add_trace(go.Scatter(x=balance_df.index, y=balance_df[column], mode='lines+markers',
                                         name=column, line=dict(color=color, width=3)))
    fig_balance.update_layout(title="資產負債表趨勢分析（季末）", **common_layout)
    charts['balance'] = fig_balance

    # 現金流量圖表（含自由現金流趨勢線）
    fig_cash = go.Figure()
    cash_colors = {'營運現金流': 'darkgreen', '投資現金流': 'goldenrod', '融資現金流': 'purple'}
    for column, color in cash_colors.items():
        fig_cash.add_trace(go.Bar(x=cash_df.index, y=cash_df[column], name=column, marker_color=color))
    fig_cash.add_trace(go.Scatter(x=cash_df.index, y=cash_df['自由現金流'], mode='lines+markers',
                                  name='自由現金流', line=dict(color='steelblue', width=3)))
    fig_cash.update_layout(title="現金流量分析（單季）", **common_layout)
    charts['cash'] = fig_cash

    return charts

def create_fscore_pie_chart(fscore_result):
    """創建 F-Score 通過率圓餅圖"""

    passed = fscore_result['total_score']
    insufficient = 9 - fscore_result['evaluated_count']
    failed = 9 - passed - insufficient

    labels, values, colors = ['通過', '未通過'], [passed, failed], ['green', 'red']
    if insufficient:
        labels.append('資料不足')
        values.append(insufficient)
        colors.append('lightgray')

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.3)])
    fig.update_layout(title=f"F-Score 通過率 ({passed}/9)", template='plotly_white', height=400)
    return fig

def create_zscore_gauge(zscore_result):
    """創建 Z-Score 風險儀表盤"""

    z_score = zscore_result['z_score']
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=z_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Altman Z-Score"},
        delta={'reference': 2.99},
        gauge={
            'axis': {'range': [0, max(5, z_score * 1.1)]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1.81], 'color': "red"},
                {'range': [1.81, 2.99], 'color': "yellow"},
                {'range': [2.99, max(5, z_score * 1.1)], 'color': "green"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': z_score},
        }
    ))
    fig.update_layout(template='plotly_white', height=400)
    return fig

# =====================================================================
# AI 分析模組
# =====================================================================

def analyze_with_openai(annual_views, fscore_result, zscore_result, dupont_result, cashflow_result,
                        quality_report, stock_info, openai_api_key, model):
    """使用 OpenAI 進行台股財務分析"""

    if not openai_api_key:
        return "請提供OpenAI API金鑰以進行AI分析"

    try:
        client = OpenAI(api_key=openai_api_key)

        # 最新年度數據（移除內部用欄位）
        latest_data = {k: v for k, v in annual_views[0].items() if k != 'period_end'} if annual_views else {}

        analysis_data = {
            "資料說明": "金額單位為新台幣元；損益與現金流為近四季合計（TTM），資產負債為期末數；年度比較為與一年前同期比較",
            "公司基本資訊": stock_info,
            "財務數據品質報告": quality_report,
            "Piotroski F-Score結果": fscore_result,
            "Altman Z-Score結果": zscore_result,
            "杜邦分析結果": dupont_result,
            "現金流分析結果": cashflow_result,
            "最新年度財務數據（近四季）": latest_data,
        }

        system_message = """你是一位專精台股財務分析和台灣會計準則的專業分析師，具備以下專業能力：

1. 深度理解台股市場特性和台灣企業經營環境
2. 熟悉台灣會計準則(TIFRS)與國際準則的差異
3. 精通四階段財務分析方法的應用和解讀
4. 了解FinMind開源資料的特點和限制性
5. 擅長客觀分析和教育性解說

你的分析目標：
- 基於已計算完成的四階段分析結果進行專業解讀，不要重新計算
- 提供客觀、教育性的財務健康診斷
- 說明資料來源限制和計算欄位的影響
- 數值標示為 null 或「資料不足」的項目，必須說明無法判斷，不可自行推測數值
- 不得杜撰資料中沒有的具體事件、日期、數字或政策
- 避免提供投資建議，專注於教育性分析
- 使用繁體中文回答"""

        user_prompt = f"""請對以下台股公司進行專業財務分析：

## 分析數據
{json.dumps(analysis_data, ensure_ascii=False, indent=2, default=str)}

## 分析要求
請進行四階段財務分析解讀，並提供以下結構化分析：

### 1. 資料來源與限制說明
- 說明FinMind開源資料特點，以及本系統採用近四季合計（TTM）的分析基準
- 標註哪些指標為 FinMind 直接提供、哪些為計算推估（股數、利息費用、市值、長期負債）
- 解釋計算推估可能造成的誤差與對分析結果的影響

### 2. Piotroski F-Score 深度解讀
- 分析9項指標的意義（若有資料不足的項目，請說明影響）
- 解讀各類別得分狀況

### 3. Altman Z-Score 風險評估
- 解讀風險等級判斷（若無法計算，請說明原因）
- 分析各組成要素影響

### 4. 杜邦分析趨勢洞察
- 分析ROE三因子變化趨勢
- 識別ROE變化的主要驅動因子

### 5. 現金流結構分析
- 評估現金流品質
- 分析資本支出模式
- 檢視獲利品質一致性

### 6. 綜合財務健康診斷
請輸出四階段評分總結表格：
| 分析階段 | 評分狀態 | 評價 | 主要發現 |

### 7. 台股產業背景（一般性說明）
- 僅能提供該產業的一般性背景知識，並在段落開頭標註「本段為一般性背景說明，非依據本系統資料」
- 不得杜撰具體事件、日期、數字或政策內容

### 8. 分析結論
- **主要優勢**：3-5個關鍵優勢
- **風險因素**：需關注的風險點
- **後續追蹤重點**：值得持續觀察的關鍵指標
- **財報綜合評比**：從營運績效、財務結構、現金流量三方面總結

請確保分析客觀專業，強調教育用途，避免投資建議。"""

        params = {
            'model': model,
            'messages': [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        }
        # o 系列推理模型不支援 temperature，且需使用 max_completion_tokens（含推理用 token）
        if model.startswith('o'):
            params['max_completion_tokens'] = 16000
        else:
            params['max_tokens'] = 3000
            params['temperature'] = 0.1

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if not content:
            return "AI 未回傳內容，可能是輸出長度不足，請稍後再試或改用其他模型。"
        return content

    except Exception as e:
        return (f"AI分析時發生錯誤：{str(e)}\n\n"
                f"建議：確認 OpenAI API 金鑰是否正確、帳戶是否有餘額；"
                f"若錯誤訊息與模型有關，可在側邊欄改選其他 AI 模型後重新分析。")

# =====================================================================
# 主應用程式
# =====================================================================

def compute_basic_info(ticker, stock_info_data, key_metrics_data, price_data, annual_views):
    """整理公司基本資訊：公司與產業 / 最新收盤價 / 估算市值與本益比"""
    
    if stock_info_data:
        company_name = stock_info_data[0].get('stock_name', ticker)
        industries = list(dict.fromkeys(item.get('industry_category', '') for item in stock_info_data if item.get('industry_category')))
        market = {'twse': '上市', 'tpex': '上櫃', 'emerging': '興櫃'}.get(stock_info_data[0].get('type'), 'N/A')
    else:
        company_name, industries, market = ticker, [], 'N/A'
    
    info = {
        'company_name': company_name,
        'industries': industries,
        'market': market,
        'close_text': 'N/A',
        'close_delta': None,
        'price_date_text': '',
        'market_cap_text': 'N/A',
        'per_text': 'N/A',
        'metrics_date_text': '',
    }
    
    # 最新收盤價與漲跌（FinMind 資料依日期升冪排列，最新一筆在最後）
    prices = sorted(price_data, key=lambda x: x['date'])
    if prices and prices[-1].get('close'):
        latest_close = prices[-1]['close']
        info['close_text'] = f"{latest_close:,.2f} 元"
        info['price_date_text'] = f"資料日期：{prices[-1]['date']}"
        if len(prices) >= 2 and prices[-2].get('close'):
            change = latest_close - prices[-2]['close']
            info['close_delta'] = f"{change:+.2f}（{change / prices[-2]['close']:+.2%}）"
    
    # 估算市值 = 最新 PBR × 最新一季歸屬母公司權益
    latest_metrics = max(key_metrics_data, key=lambda x: x['date']) if key_metrics_data else {}
    if latest_metrics.get('PBR') and annual_views:
        equity = annual_views[0]['equity_parent'] or annual_views[0]['stockholdersequity']
        if not is_missing(equity):
            info['market_cap_text'] = format_large_number(latest_metrics['PBR'] * equity)
    if latest_metrics.get('PER'):
        info['per_text'] = f"{latest_metrics['PER']:.2f}"
    if latest_metrics:
        info['metrics_date_text'] = f"資料日期：{latest_metrics.get('date')}"
    
    return info

def show_basic_info(ticker, info):
    """三欄式公司基本資訊"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"{info['company_name']} ({ticker})")
        st.write(f"**產業類別:** {'、'.join(info['industries']) if info['industries'] else 'N/A'}")
        st.write(f"**市場別:** {info['market']}")
    
    with col2:
        st.metric("最新收盤價", info['close_text'], delta=info['close_delta'])
        if info['price_date_text']:
            st.caption(info['price_date_text'])
    
    with col3:
        st.metric("估算市值", info['market_cap_text'])
        st.metric("本益比 (PER)", info['per_text'])
        if info['metrics_date_text']:
            st.caption(f"市值＝PBR×歸屬母公司權益（估算）；{info['metrics_date_text']}")


def show_fourstage_analysis(quality_report, fscore_result, zscore_result, dupont_result, cashflow_result):
    """四階段財報分析頁籤"""

    with st.expander("📊 數據品質報告", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("數據完整性", quality_report["數據完整性"])
            st.metric("財報季數", f"{quality_report['財報季數']} 季")
            st.metric("最新財報季度", quality_report["最新財報季度"])
        with col2:
            if quality_report["缺失欄位"]:
                st.warning("⚠️ 缺失欄位：")
                for field in quality_report["缺失欄位"]:
                    st.write(f"- {field}")
            for warning in quality_report["數據警告"]:
                st.warning(warning)
        st.info(f"ℹ️ 分析基準：{quality_report['分析基準']}")
        st.markdown("**資料轉換說明**")
        for note in quality_report["資料轉換說明"]:
            st.write(f"- {note}")
        st.markdown("**計算欄位說明**")
        for explanation in quality_report["計算欄位說明"]:
            st.write(f"- {explanation}")

    # 階段一：Piotroski F-Score
    st.markdown("### 📈 階段一：Piotroski F-Score")
    if fscore_result:
        st.caption(f"比較基準：{fscore_result['current_date']} vs {fscore_result['previous_date']}")
        col1, col2 = st.columns([2, 1])
        with col1:
            total_score = fscore_result['total_score']
            rating = '優秀' if total_score >= 7 else '良好' if total_score >= 5 else '需關注'
            insufficient = 9 - fscore_result['evaluated_count']
            st.metric("F-Score 總分", f"{total_score}/9（{rating}）",
                      help="Piotroski F-Score 總分" + (f"；{insufficient} 項資料不足，以 0 分計" if insufficient else ""))
            if insufficient:
                st.warning(f"有 {insufficient} 項指標因資料不足無法判斷，以 0 分計算")
            for title, key in [("獲利能力指標 (4項)", 'profitability_scores'),
                               ("槓桿與流動性指標 (3項)", 'leverage_scores'),
                               ("營運效率指標 (2項)", 'efficiency_scores')]:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(fscore_result[key]), use_container_width=True, hide_index=True)
        with col2:
            st.plotly_chart(create_fscore_pie_chart(fscore_result), use_container_width=True)
    else:
        st.warning("財務數據不足，無法計算 Piotroski F-Score（需要至少 8 季數據，請將起始日期提前）")

    st.markdown("---")

    # 階段二：Altman Z-Score
    st.markdown("### ⚖️ 階段二：Altman Z-Score")
    if zscore_result:
        st.caption(f"計算基準：{zscore_result['date']}")
        col1, col2 = st.columns([2, 1])
        with col1:
            z_score = zscore_result['z_score']
            st.metric("Altman Z-Score", "N/A" if z_score is None else f"{z_score:.2f}")
            st.metric("風險等級", f"{zscore_result['risk_level']} {zscore_result['risk_emoji']}")
            if zscore_result['missing_components']:
                st.warning(f"缺少資料無法計算：{'、'.join(zscore_result['missing_components'])}")

            st.markdown("**Z-Score 組成要素**")
            st.dataframe(pd.DataFrame([
                {'項目': f"{key}項", '描述': comp['description'],
                 '比率值': format_ratio(comp['ratio']), '權重後數值': format_ratio(comp['weighted'])}
                for key, comp in zscore_result['components'].items()
            ]), use_container_width=True, hide_index=True)

            st.markdown("**計算基礎數據**")
            base = zscore_result['base_data']
            st.dataframe(pd.DataFrame([
                {'項目': '營運資本', '金額': format_large_number(base['working_capital'])},
                {'項目': '總資產', '金額': format_large_number(base['total_assets'])},
                {'項目': '保留盈餘', '金額': format_large_number(base['retained_earnings'])},
                {'項目': 'EBIT（近四季）', '金額': format_large_number(base['ebit'])},
                {'項目': '市值（估算）', '金額': format_large_number(base['market_cap'])},
                {'項目': '總負債', '金額': format_large_number(base['total_liabilities'])},
                {'項目': '營收（近四季）', '金額': format_large_number(base['revenues'])},
            ]), use_container_width=True, hide_index=True)
        with col2:
            if zscore_result['z_score'] is not None:
                st.plotly_chart(create_zscore_gauge(zscore_result), use_container_width=True)
    else:
        st.warning("無法計算 Altman Z-Score")

    st.markdown("---")

    # 階段三：杜邦分析
    st.markdown("### 🔍 階段三：杜邦分析")
    if dupont_result['annual_data']:
        st.metric("當前 ROE（近四季）", format_percent(dupont_result['annual_data'][0]['direct_roe']))
        st.markdown("**年度杜邦分析**")
        st.dataframe(pd.DataFrame([
            {'期間': item['date'], '淨利率': format_ratio(item['net_margin']),
             '資產周轉率': format_ratio(item['asset_turnover']), '權益乘數': format_ratio(item['equity_multiplier']),
             '計算ROE': format_ratio(item['calculated_roe']), '直接ROE': format_ratio(item['direct_roe'])}
            for item in dupont_result['annual_data']
        ]), use_container_width=True, hide_index=True)
        if dupont_result['trends']:
            st.markdown("**趨勢變化分析（與一年前比較）**")
            st.dataframe(pd.DataFrame([
                {'因子': item['factor'], '變化': format_ratio(item['change'])}
                for item in dupont_result['trends']
            ]), use_container_width=True, hide_index=True)
    else:
        st.warning("無法計算杜邦分析")

    st.markdown("---")

    # 階段四：現金流分析
    st.markdown("### 💰 階段四：現金流分析")
    if cashflow_result:
        st.caption(f"計算基準：{cashflow_result['date']}")
        ratio = cashflow_result['cf_quality_ratio']
        st.metric("現金流品質比率", format_ratio(ratio, 2),
                  delta=f"{cashflow_result['quality_assessment']} {cashflow_result['quality_emoji']}",
                  delta_color="off" if ratio is None else "normal")

        st.markdown("**現金流關鍵指標**")
        fcf = cashflow_result['free_cashflow']
        st.dataframe(pd.DataFrame([
            {'指標': '營運現金流品質比率', '數值': format_ratio(ratio, 2), '評估': cashflow_result['quality_assessment']},
            {'指標': '自由現金流', '數值': format_large_number(fcf),
             '評估': 'N/A' if fcf is None else ('正值為佳' if fcf > 0 else '需關注')},
        ]), use_container_width=True, hide_index=True)

        st.markdown("**現金流結構分析**")
        st.dataframe(pd.DataFrame([
            {'類型': item['type'], '金額': format_large_number(item['amount'])}
            for item in cashflow_result['structure_analysis']
        ]), use_container_width=True, hide_index=True)

        st.markdown("**詳細現金流數據（近四季）**")
        detail = cashflow_result['detailed_data']
        st.dataframe(pd.DataFrame([
            {'項目': '營運現金流', '金額': format_large_number(detail['operating_cf'])},
            {'項目': '投資現金流', '金額': format_large_number(detail['investing_cf'])},
            {'項目': '融資現金流', '金額': format_large_number(detail['financing_cf'])},
            {'項目': '淨利潤', '金額': format_large_number(detail['net_income'])},
            {'項目': '資本支出', '金額': format_large_number(detail['capex'])},
            {'項目': '現金流總計', '金額': format_large_number(detail['total_cf'])},
        ]), use_container_width=True, hide_index=True)
    else:
        st.warning("無法計算現金流分析")

def build_summary_rows(fscore_result, zscore_result, dupont_result, cashflow_result):
    """分析數據摘要（網頁與 PDF 共用）"""
    summary_data = []
    if fscore_result:
        score = fscore_result['total_score']
        summary_data.append({'分析項目': 'Piotroski F-Score', '結果': f"{score}/9",
                             '狀態': '優秀' if score >= 7 else '良好' if score >= 5 else '需關注'})
    if zscore_result:
        z = zscore_result['z_score']
        summary_data.append({'分析項目': 'Altman Z-Score', '結果': 'N/A' if z is None else f"{z:.2f}",
                             '狀態': zscore_result['risk_level']})
    if dupont_result['annual_data']:
        roe = dupont_result['annual_data'][0]['direct_roe']
        status = 'N/A' if roe is None else '優秀' if roe > 0.15 else '良好' if roe > 0.10 else '需關注'
        summary_data.append({'分析項目': 'ROE (杜邦分析，近四季)', '結果': format_percent(roe), '狀態': status})
    if cashflow_result:
        summary_data.append({'分析項目': '現金流品質', '結果': format_ratio(cashflow_result['cf_quality_ratio'], 2),
                             '狀態': cashflow_result['quality_assessment']})
    return summary_data

def run_analysis(ticker, finmind_api_token, openai_api_key, ai_model, start_date):
    """
    執行完整分析流程，回傳結果 dict（失敗時回傳 None）
    結果會存入 session_state，按下「匯出 PDF」等按鈕重新整理頁面時不會遺失
    """
    is_valid, error_msg = validate_taiwan_stock_code(ticker)
    if not is_valid:
        st.error(error_msg)
        return None
    if not finmind_api_token:
        st.warning("請輸入 FinMind API Token")
        return None
    
    try:
        # 取得與處理數據
        with st.spinner(f"正在從 FinMind API 獲取 {ticker} 的財務報表資料..."):
            raw_data = get_finmind_raw_data(ticker, finmind_api_token, start_date)
            quarterly, conversion_notes = build_quarterly_data(raw_data)
            annual_views = build_annual_views(quarterly, raw_data.get('key_metrics', []))
        
        stock_info_data = raw_data.get('stock_info', [])
        is_financial = is_financial_industry(stock_info_data)
        basic_info = compute_basic_info(ticker, stock_info_data, raw_data.get('key_metrics', []),
                                        raw_data.get('stock_price', []), annual_views)
        
        # 展示用資料
        income_df, balance_df, ratio_df, cash_df = build_display_tables(quarterly)
        if income_df.empty:
            st.error("無法獲取有效的財務數據")
            return None
        
        # 四階段分析
        quality_report = analyze_data_quality(quarterly, annual_views, conversion_notes, is_financial)
        fscore_result = calculate_piotroski_fscore(annual_views)
        zscore_result = calculate_altman_zscore(annual_views)
        dupont_result = calculate_dupont_analysis(annual_views)
        cashflow_result = calculate_cashflow_analysis(annual_views)
        
        # AI 分析
        ai_analysis = None
        if openai_api_key:
            stock_info = {
                'company_name': basic_info['company_name'],
                'stock_code': ticker,
                'industry': '、'.join(basic_info['industries']) if basic_info['industries'] else 'N/A',
                'market': basic_info['market'],
                'is_financial_industry': is_financial,
            }
            with st.spinner("正在使用AI進行四階段財務分析..."):
                ai_analysis = analyze_with_openai(
                    annual_views, fscore_result, zscore_result, dupont_result, cashflow_result,
                    quality_report, stock_info, openai_api_key, ai_model
                )
        
        return {
            'ticker': ticker,
            'company_name': basic_info['company_name'],
            'industries': basic_info['industries'],
            'market': basic_info['market'],
            'basic_info': basic_info,
            'is_financial': is_financial,
            'quality_report': quality_report,
            'fscore_result': fscore_result,
            'zscore_result': zscore_result,
            'dupont_result': dupont_result,
            'cashflow_result': cashflow_result,
            'summary_rows': build_summary_rows(fscore_result, zscore_result, dupont_result, cashflow_result),
            'income_df': income_df,
            'balance_df': balance_df,
            'ratio_df': ratio_df,
            'cash_df': cash_df,
            'ai_analysis': ai_analysis,
            'ai_model': ai_model,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
    
    except Exception as e:
        st.error(f"分析過程中發生錯誤：{str(e)}")
        st.info("請確認股票代碼與 FinMind API Token 是否正確，或稍後再試。")
        # 詳細錯誤只寫入伺服器記錄（Streamlit Cloud 的 Manage app → Logs），不顯示在公開網頁上
        print(traceback.format_exc())
        return None

def get_pdf_bytes(result):
    """產生 PDF（同一份分析結果只產生一次）"""
    if 'pdf_bytes' not in result:
        try:
            from pdf_report import generate_pdf_report
        except ImportError:
            result['pdf_bytes'] = None
            result['pdf_error'] = "缺少 reportlab 套件，請執行 pip install reportlab 後重新啟動"
            return None
        
        cashflow = result['cashflow_result']
        report = dict(result)
        if cashflow:
            detail = cashflow['detailed_data']
            report['cashflow_detail_rows'] = [
                ('自由現金流', format_large_number(cashflow['free_cashflow'])),
                ('營運現金流', format_large_number(detail['operating_cf'])),
                ('投資現金流', format_large_number(detail['investing_cf'])),
                ('融資現金流', format_large_number(detail['financing_cf'])),
                ('淨利潤', format_large_number(detail['net_income'])),
                ('資本支出', format_large_number(detail['capex'])),
            ]
        try:
            result['pdf_bytes'] = generate_pdf_report(report)
        except Exception as e:
            result['pdf_bytes'] = None
            result['pdf_error'] = f"PDF 產生失敗：{e}"
            print(traceback.format_exc())
    return result['pdf_bytes']

def show_export_button(result):
    """「匯出」按鈕：下載 PDF，存檔位置由瀏覽器詢問"""
    with st.spinner("正在產生 PDF 報告..."):
        pdf_bytes = get_pdf_bytes(result)
    if pdf_bytes is None:
        st.error(result.get('pdf_error', 'PDF 產生失敗'))
        return
    
    file_name = f"{result['ticker']}_{result['company_name']}_財報分析_{datetime.now().strftime('%Y%m%d')}.pdf"
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            "📄 匯出 PDF",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.caption("💡 若希望每次下載時選擇存檔位置，請在瀏覽器設定中開啟「下載前詢問每個檔案的儲存位置」"
                   "（Chrome／Edge：設定 → 下載；Safari：設定 → 一般 → 檔案下載位置選「每次都詢問」）。")

def display_results(result):
    """顯示分析結果（從 session_state 讀取，頁面重新整理時不需重新分析）"""
    
    show_basic_info(result['ticker'], result['basic_info'])
    if result['is_financial']:
        st.warning("⚠️ 此公司屬於金融保險業，財報結構與一般產業不同，F-Score 與 Z-Score 不適用，結果僅供參考。")
    
    show_export_button(result)
    
    income_df, balance_df, ratio_df, cash_df = result['income_df'], result['balance_df'], result['ratio_df'], result['cash_df']
    charts = create_financial_charts(income_df, balance_df, cash_df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "損益表分析", "資產負債表分析", "現金流量表分析", "四階段財報分析", "AI分析"
    ])
    
    with tab1:
        st.subheader("損益表分析")
        st.plotly_chart(charts['income'], use_container_width=True)
        st.subheader("完整損益表（單季，單位：億元）")
        st.dataframe(descending(income_df), use_container_width=True)
    
    with tab2:
        st.subheader("資產負債表分析")
        st.plotly_chart(charts['balance'], use_container_width=True)
        st.subheader("財務比率")
        st.dataframe(descending(ratio_df), use_container_width=True)
        st.subheader("完整資產負債表（季末，單位：億元）")
        st.dataframe(descending(balance_df), use_container_width=True)
    
    with tab3:
        st.subheader("現金流量表分析")
        st.plotly_chart(charts['cash'], use_container_width=True)
        st.subheader("完整現金流量表（單季，單位：億元）")
        st.dataframe(descending(cash_df), use_container_width=True)
    
    with tab4:
        st.subheader("四階段財報分析")
        show_fourstage_analysis(result['quality_report'], result['fscore_result'], result['zscore_result'],
                                result['dupont_result'], result['cashflow_result'])
    
    with tab5:
        st.subheader("AI 綜合財務分析")
        if result['ai_analysis']:
            st.markdown("### 🎯 AI 財務分析報告")
            st.markdown(result['ai_analysis'])
        else:
            st.warning("請在側邊欄輸入 OpenAI API 金鑰以使用 AI 分析功能")
            st.info("💡 AI分析功能需要OpenAI API金鑰，請在左側邊欄輸入後重新分析")
        
        st.markdown("### 📋 分析數據摘要")
        if result['summary_rows']:
            st.dataframe(pd.DataFrame(result['summary_rows']), use_container_width=True, hide_index=True)
    
    st.success(f"✅ 分析完成！（{result['generated_at']}）")

def main():
    # 側邊欄
    st.sidebar.header("Code Gym", divider="rainbow")
    
    # 登出按鈕（同時清除分析結果）
    if st.sidebar.button("🚪 登出"):
        st.session_state.clear()
        st.rerun()
    
    ticker = st.sidebar.text_input(
        "輸入台股代碼（例如：2330 代表台積電）",
        "2330",
        help="請輸入四位數字的台股代碼，例如：2330、2454、2317、2412"
    ).strip()
    
    finmind_api_token = st.sidebar.text_input(
        "輸入 FinMind API Token",
        type="password",
        value="",
        help="請前往 FinMind 官網（finmindtrade.com）註冊後，於使用者資訊頁取得 API Token"
    )
    
    openai_api_key = st.sidebar.text_input(
        "輸入OpenAI API金鑰",
        type="password",
        value="",
        help="用於AI財務分析功能"
    )
    
    ai_model = st.sidebar.selectbox(
        "AI 模型",
        AI_MODEL_OPTIONS,
        index=0,
        help="o4-mini 為規格指定模型；若帳戶無法使用，可改選 gpt-4o-mini"
    )
    
    # 起始日期：預設為 5 年前（年對年比較至少需要 8 季、杜邦三年分析需要 12 季）
    default_start_date = datetime.now() - timedelta(days=5 * 365)
    start_date = st.sidebar.date_input(
        "數據起始日期",
        value=default_start_date,
        help="選擇財務數據的起始日期（預設為5年前，建議至少 3 年以上）"
    ).strftime('%Y-%m-%d')
    
    # 免責聲明
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📢 免責聲明
    本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
    請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
    """)
    
    if st.sidebar.button("分析股票", type="primary"):
        result = run_analysis(ticker, finmind_api_token, openai_api_key, ai_model, start_date)
        if result:
            st.session_state['analysis_result'] = result
        else:
            # 分析失敗時清除舊結果，避免誤以為是本次的分析
            st.session_state.pop('analysis_result', None)
    
    result = st.session_state.get('analysis_result')
    if result:
        display_results(result)

if __name__ == "__main__":
    main()
