import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import traceback
import requests
from openai import OpenAI
from datetime import datetime, timedelta
import numpy as np
import re

# 設置頁面配置
st.set_page_config(page_title="AI 分析台股基本面應用", layout="wide")
st.header("【Code Gym】AI 分析台股基本面應用", divider="rainbow")

# 函數：格式化大數字
def format_large_number(num):
    """將大數字轉換為易讀格式"""
    if pd.isna(num) or num == 0:
        return "0"
    if abs(num) >= 1e12:
        return f"{num/1e12:.2f}兆"
    elif abs(num) >= 1e9:
        return f"{num/1e9:.2f}億" 
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}百萬"
    else:
        return f"{num:,.0f}"

# 函數：驗證台股代碼格式
def validate_taiwan_stock_code(stock_code):
    """驗證台股代碼是否為四位數字格式"""
    if not stock_code:
        return False, "請輸入股票代碼"
    
    # 去除空格
    stock_code = stock_code.strip()
    
    # 檢查是否為4位數字
    if not re.match(r'^\d{4}$', stock_code):
        return False, "台股代碼必須為四位數字格式（例如：2330）"
    
    return True, ""

# 函數：從FinMind API獲取財務數據
def get_finmind_data_from_apis(stock_id, api_token, start_date="2019-01-01"):
    """從FinMind API獲取完整台股財務數據"""
    try:
        # FinMind API統一端點
        base_url = "https://api.finmindtrade.com/api/v4/data"
        
        # 數據集配置
        datasets = {
            'financial_statements': 'TaiwanStockFinancialStatements',
            'balance_sheet': 'TaiwanStockBalanceSheet', 
            'cash_flow': 'TaiwanStockCashFlowsStatement',
            'stock_info': 'TaiwanStockInfo',
            'key_metrics': 'TaiwanStockPER'
        }
        
        finmind_data = {}
        
        st.info(f"正在從FinMind API獲取 {stock_id} 的財務報表資料...")
        
        # 獲取各類財務數據
        for data_type, dataset in datasets.items():
            params = {
                'dataset': dataset,
                'data_id': stock_id,
                'start_date': start_date,
                'token': api_token
            }
            
            # 對於關鍵指標，使用較短的日期範圍避免過多數據
            if data_type == 'key_metrics':
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                params['end_date'] = end_date
            
            response = requests.get(base_url, params=params)
            
            if response.status_code != 200:
                raise Exception(f"{data_type} API請求失敗: {response.status_code}")
            
            data = response.json()
            
            if data.get('status') != 200:
                raise Exception(f"{data_type} API回傳錯誤: {data.get('msg', '未知錯誤')}")
            
            finmind_data[data_type] = data.get('data', [])
        
        # 驗證數據完整性
        if not finmind_data['financial_statements']:
            raise Exception("無法獲取損益表數據，請檢查股票代碼是否正確")
        
        # 將FinMind格式轉換為內部標準格式
        converted_data = convert_finmind_to_standard_format(finmind_data)
        
        return converted_data
        
    except Exception as e:
        raise Exception(f"獲取FinMind數據時發生錯誤: {str(e)}")

# 函數：FinMind欄位對應轉換
def convert_finmind_to_standard_format(finmind_data):
    """將FinMind API數據轉換為內部標準格式"""
    
    # FinMind type欄位到內部標準欄位的對應表
    field_mapping = {
        # 損益表欄位對應
        'Revenue': 'revenues',
        'GrossProfit': 'grossprofit', 
        'OperatingIncome': 'operatingincomeloss',
        'IncomeAfterTaxes': 'netincomeloss',
        'PreTaxIncome': 'incomelossfromcontinuingoperationsbeforeincometaxes',
        'EPS': 'eps_for_calculation',
        'TotalNonoperatingIncomeAndExpense': 'nonoperating_income_expense',
        
        # 資產負債表欄位對應
        'TotalAssets': 'assets',
        'Liabilities': 'liabilities',
        'Equity': 'stockholdersequity', 
        'CurrentAssets': 'assetscurrent',
        'CurrentLiabilities': 'liabilitiescurrent',
        'RetainedEarnings': 'retainedearningsaccumulateddeficit',
        'NoncurrentLiabilities': 'longtermdebtnoncurrent',
        
        # 現金流量表欄位對應
        'CashFlowsFromOperatingActivities': 'netcashprovidedbyusedinoperatingactivities',
        'CashProvidedByInvestingActivities': 'netcashprovidedbyusedininvestingactivities', 
        'CashFlowsProvidedFromFinancingActivities': 'netcashprovidedbyusedinfinancingactivities',
        'PropertyAndPlantAndEquipment': 'capital_expenditure_raw'
    }
    
    # 組織數據按日期分組
    data_by_date = {}
    
    # 處理財務報表數據（損益表）
    for item in finmind_data.get('financial_statements', []):
        date = item['date']
        type_name = item['type'] 
        value = item['value']
        
        if date not in data_by_date:
            data_by_date[date] = {
                'date': date,
                'stock_id': item['stock_id']
            }
        
        # 根據對應表轉換欄位名稱
        if type_name in field_mapping:
            standard_field = field_mapping[type_name]
            data_by_date[date][standard_field] = value
    
    # 處理資產負債表數據
    for item in finmind_data.get('balance_sheet', []):
        date = item['date']
        type_name = item['type']
        value = item['value']
        
        if date not in data_by_date:
            data_by_date[date] = {
                'date': date,
                'stock_id': item['stock_id']
            }
        
        if type_name in field_mapping:
            standard_field = field_mapping[type_name]
            data_by_date[date][standard_field] = value
    
    # 處理現金流量表數據
    for item in finmind_data.get('cash_flow', []):
        date = item['date']
        type_name = item['type']
        value = item['value']
        
        if date not in data_by_date:
            data_by_date[date] = {
                'date': date, 
                'stock_id': item['stock_id']
            }
        
        if type_name in field_mapping:
            standard_field = field_mapping[type_name]
            data_by_date[date][standard_field] = value
    
    # 轉換為列表並按日期降序排序
    financial_data = list(data_by_date.values())
    financial_data.sort(key=lambda x: x['date'], reverse=True)
    
    # 實施計算補償機制
    financial_data = apply_calculation_compensation(financial_data, finmind_data)
    
    return {
        'financial_statements': financial_data,
        'stock_info': finmind_data.get('stock_info', []),
        'key_metrics': finmind_data.get('key_metrics', []),
        'raw_finmind_data': finmind_data
    }

# 函數：實施計算補償機制
def apply_calculation_compensation(financial_data, finmind_data):
    """實施缺失欄位的計算補償機制"""
    
    # 獲取PBR數據用於市值計算
    pbr_data = {}
    for item in finmind_data.get('key_metrics', []):
        date = item['date']
        if 'PBR' in item:
            pbr_data[date] = item['PBR']
    
    for data_item in financial_data:
        date = data_item['date']
        
        # 1. 計算加權平均股數
        if 'netincomeloss' in data_item and 'eps_for_calculation' in data_item:
            net_income = data_item['netincomeloss']
            eps = data_item['eps_for_calculation']
            
            if eps and eps != 0:
                # 加權平均股數 = 淨利潤 ÷ EPS (轉換為千股)
                weighted_avg_shares = (net_income / eps) / 1000
                data_item['weightedaveragenumberofsharesoutstandingbasic'] = weighted_avg_shares
        
        # 2. 推估利息費用
        if 'nonoperating_income_expense' in data_item:
            nonoperating = data_item['nonoperating_income_expense']
            if nonoperating < 0:
                # 如果營業外為支出（負值），取絕對值作為利息費用
                data_item['interestexpensenonoperating'] = abs(nonoperating)
            else:
                data_item['interestexpensenonoperating'] = 0
        
        # 3. 計算市值（用於Altman Z-Score）
        if 'stockholdersequity' in data_item and date in pbr_data:
            equity = data_item['stockholdersequity']
            pbr = pbr_data[date]
            if equity and pbr:
                # 市值 = PBR × 股東權益
                market_cap = pbr * equity
                data_item['market_capitalization'] = market_cap
        
        # 4. 處理資本支出（取絕對值）
        if 'capital_expenditure_raw' in data_item:
            capex_raw = data_item['capital_expenditure_raw']
            data_item['paymentstoacquireproductiveassets'] = abs(capex_raw) if capex_raw else 0
    
    return financial_data

# 函數：數據品質檢查
def analyze_data_quality(financial_data, finmind_data):
    """分析財務數據品質並生成報告"""
    
    quality_report = {
        "數據完整性": "良好",
        "數據年份": len(financial_data),
        "缺失欄位": [],
        "數據警告": [],
        "計算欄位說明": []
    }
    
    # 檢查關鍵欄位
    required_fields = [
        'revenues', 'netincomeloss', 'assets', 'stockholdersequity',
        'netcashprovidedbyusedinoperatingactivities'
    ]
    
    for field in required_fields:
        missing_years = []
        for year_data in financial_data:
            if not year_data.get(field) and year_data.get(field) != 0:
                missing_years.append(year_data.get('date', '未知年份'))
        
        if missing_years:
            quality_report["缺失欄位"].append(f"{field}: {', '.join(missing_years)}")
    
    # 說明計算欄位
    if any('weightedaveragenumberofsharesoutstandingbasic' in item for item in financial_data):
        quality_report["計算欄位說明"].append(
            "加權平均股數：由淨利潤÷EPS計算得出，為近似值"
        )
    
    if any('interestexpensenonoperating' in item for item in financial_data):
        quality_report["計算欄位說明"].append(
            "利息費用：由營業外收支推估，如為支出則取絕對值"
        )
    
    if any('market_capitalization' in item for item in financial_data):
        quality_report["計算欄位說明"].append(
            "市值：由PBR×股東權益計算，用於Z-Score分析"
        )
    
    # 檢查數據年份
    if len(financial_data) < 2:
        quality_report["數據警告"].append("財務數據少於2年，無法進行年度比較分析")
        quality_report["數據完整性"] = "嚴重不足"
    elif quality_report["缺失欄位"]:
        quality_report["數據完整性"] = "部分缺失"
    
    return quality_report

# 函數：計算Piotroski F-Score
def calculate_piotroski_fscore(financial_data):
    """計算Piotroski F-Score (9項指標)"""
    
    if len(financial_data) < 2:
        return None
    
    current = financial_data[0]  # 最新年度
    previous = financial_data[1]  # 前一年度
    
    scores = {
        'profitability_scores': [],
        'leverage_scores': [],
        'efficiency_scores': [],
        'total_score': 0
    }
    
    # 獲利能力指標（4項）
    # 1. ROA正值檢查
    current_roa = current.get('netincomeloss', 0) / current.get('assets', 1) if current.get('assets') else 0
    roa_positive = 1 if current_roa > 0 else 0
    scores['profitability_scores'].append({
        'description': 'ROA正值檢查',
        'current_value': f"{current_roa:.4f}",
        'score': roa_positive,
        'status': '✓' if roa_positive else '✗'
    })
    
    # 2. 營運現金流正值檢查
    operating_cf = current.get('netcashprovidedbyusedinoperatingactivities', 0)
    cf_positive = 1 if operating_cf > 0 else 0
    scores['profitability_scores'].append({
        'description': '營運現金流正值檢查',
        'current_value': format_large_number(operating_cf),
        'score': cf_positive,
        'status': '✓' if cf_positive else '✗'
    })
    
    # 3. ROA年增率檢查
    previous_roa = previous.get('netincomeloss', 0) / previous.get('assets', 1) if previous.get('assets') else 0
    roa_improved = 1 if current_roa > previous_roa else 0
    scores['profitability_scores'].append({
        'description': 'ROA年增率檢查',
        'current_value': f"{current_roa:.4f}",
        'previous_value': f"{previous_roa:.4f}",
        'score': roa_improved,
        'status': '✓' if roa_improved else '✗'
    })
    
    # 4. 營運現金流品質檢查
    net_income = current.get('netincomeloss', 0)
    cf_quality = 1 if operating_cf > net_income else 0
    scores['profitability_scores'].append({
        'description': '營運現金流品質檢查',
        'current_value': format_large_number(operating_cf),
        'comparison_value': format_large_number(net_income),
        'score': cf_quality,
        'status': '✓' if cf_quality else '✗'
    })
    
    # 槓桿與流動性指標（3項）
    # 5. 長期負債比率改善檢查
    current_ltd_ratio = current.get('longtermdebtnoncurrent', 0) / current.get('assets', 1) if current.get('assets') else 0
    previous_ltd_ratio = previous.get('longtermdebtnoncurrent', 0) / previous.get('assets', 1) if previous.get('assets') else 0
    ltd_improved = 1 if current_ltd_ratio < previous_ltd_ratio else 0
    scores['leverage_scores'].append({
        'description': '長期負債比率改善檢查',
        'current_value': f"{current_ltd_ratio:.4f}",
        'previous_value': f"{previous_ltd_ratio:.4f}",
        'score': ltd_improved,
        'status': '✓' if ltd_improved else '✗'
    })
    
    # 6. 流動比率改善檢查
    current_ratio_current = current.get('assetscurrent', 0) / current.get('liabilitiescurrent', 1) if current.get('liabilitiescurrent') else 0
    current_ratio_previous = previous.get('assetscurrent', 0) / previous.get('liabilitiescurrent', 1) if previous.get('liabilitiescurrent') else 0
    current_ratio_improved = 1 if current_ratio_current > current_ratio_previous else 0
    scores['leverage_scores'].append({
        'description': '流動比率改善檢查',
        'current_value': f"{current_ratio_current:.2f}",
        'previous_value': f"{current_ratio_previous:.2f}",
        'score': current_ratio_improved,
        'status': '✓' if current_ratio_improved else '✗'
    })
    
    # 7. 股份稀釋檢查
    current_shares = current.get('weightedaveragenumberofsharesoutstandingbasic', 0)
    previous_shares = previous.get('weightedaveragenumberofsharesoutstandingbasic', 0)
    shares_not_increased = 1 if current_shares <= previous_shares and previous_shares > 0 else 0
    scores['leverage_scores'].append({
        'description': '股份稀釋檢查',
        'current_value': format_large_number(current_shares),
        'previous_value': format_large_number(previous_shares),
        'score': shares_not_increased,
        'status': '✓' if shares_not_increased else '✗'
    })
    
    # 營運效率指標（2項）
    # 8. 毛利率改善檢查
    current_gross_margin = current.get('grossprofit', 0) / current.get('revenues', 1) if current.get('revenues') else 0
    previous_gross_margin = previous.get('grossprofit', 0) / previous.get('revenues', 1) if previous.get('revenues') else 0
    gross_margin_improved = 1 if current_gross_margin > previous_gross_margin else 0
    scores['efficiency_scores'].append({
        'description': '毛利率改善檢查',
        'current_value': f"{current_gross_margin:.4f}",
        'previous_value': f"{previous_gross_margin:.4f}",
        'score': gross_margin_improved,
        'status': '✓' if gross_margin_improved else '✗'
    })
    
    # 9. 資產周轉率改善檢查
    current_asset_turnover = current.get('revenues', 0) / current.get('assets', 1) if current.get('assets') else 0
    previous_asset_turnover = previous.get('revenues', 0) / previous.get('assets', 1) if previous.get('assets') else 0
    asset_turnover_improved = 1 if current_asset_turnover > previous_asset_turnover else 0
    scores['efficiency_scores'].append({
        'description': '資產周轉率改善檢查',
        'current_value': f"{current_asset_turnover:.4f}",
        'previous_value': f"{previous_asset_turnover:.4f}",
        'score': asset_turnover_improved,
        'status': '✓' if asset_turnover_improved else '✗'
    })
    
    # 計算總分
    total_score = (
        sum(item['score'] for item in scores['profitability_scores']) +
        sum(item['score'] for item in scores['leverage_scores']) +
        sum(item['score'] for item in scores['efficiency_scores'])
    )
    scores['total_score'] = total_score
    
    return scores

# 函數：計算Altman Z-Score
def calculate_altman_zscore(financial_data):
    """計算Altman Z-Score"""
    
    if not financial_data:
        return None
    
    current = financial_data[0]
    
    # 計算五個組成要素
    # A項：營運資本/總資產
    working_capital = current.get('assetscurrent', 0) - current.get('liabilitiescurrent', 0)
    total_assets = current.get('assets', 1)
    a_ratio = working_capital / total_assets if total_assets else 0
    a_weighted = a_ratio * 1.2
    
    # B項：保留盈餘/總資產
    retained_earnings = current.get('retainedearningsaccumulateddeficit', 0)
    b_ratio = retained_earnings / total_assets if total_assets else 0
    b_weighted = b_ratio * 1.4
    
    # C項：EBIT/總資產
    operating_income = current.get('operatingincomeloss', 0)
    interest_expense = current.get('interestexpensenonoperating', 0)
    ebit = operating_income + interest_expense  # 直接相加，不使用絕對值
    c_ratio = ebit / total_assets if total_assets else 0
    c_weighted = c_ratio * 3.3
    
    # D項：市值/總負債
    market_cap = current.get('market_capitalization', 0)
    total_liabilities = current.get('liabilities', 1)
    d_ratio = market_cap / total_liabilities if total_liabilities else 0
    d_weighted = d_ratio * 0.6
    
    # E項：營收/總資產
    revenues = current.get('revenues', 0)
    e_ratio = revenues / total_assets if total_assets else 0
    e_weighted = e_ratio * 1.0
    
    # 計算Z-Score
    z_score = a_weighted + b_weighted + c_weighted + d_weighted + e_weighted
    
    # 風險等級判斷
    if z_score > 2.99:
        risk_level = "安全區域"
        risk_emoji = "😊"
    elif z_score >= 1.81:
        risk_level = "灰色區域"
        risk_emoji = "😐"
    else:
        risk_level = "危險區域"
        risk_emoji = "😰"
    
    return {
        'z_score': z_score,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'components': {
            'A': {'ratio': a_ratio, 'weighted': a_weighted, 'description': '營運資本/總資產'},
            'B': {'ratio': b_ratio, 'weighted': b_weighted, 'description': '保留盈餘/總資產'},
            'C': {'ratio': c_ratio, 'weighted': c_weighted, 'description': 'EBIT/總資產'},
            'D': {'ratio': d_ratio, 'weighted': d_weighted, 'description': '市值/總負債'},
            'E': {'ratio': e_ratio, 'weighted': e_weighted, 'description': '營收/總資產'}
        },
        'base_data': {
            'working_capital': working_capital,
            'total_assets': total_assets,
            'retained_earnings': retained_earnings,
            'ebit': ebit,
            'market_cap': market_cap,
            'total_liabilities': total_liabilities,
            'revenues': revenues
        }
    }

# 函數：計算杜邦分析
def calculate_dupont_analysis(financial_data):
    """計算杜邦分析（ROE三因子分解）"""
    
    dupont_data = []
    
    # 取最近3年數據
    for i, data in enumerate(financial_data[:3]):
        date = data['date']
        
        # 計算三因子
        net_income = data.get('netincomeloss', 0)
        revenues = data.get('revenues', 1)
        assets = data.get('assets', 1)
        equity = data.get('stockholdersequity', 1)
        
        # 淨利率
        net_margin = net_income / revenues if revenues else 0
        
        # 資產周轉率
        asset_turnover = revenues / assets if assets else 0
        
        # 權益乘數
        equity_multiplier = assets / equity if equity else 0
        
        # 計算ROE
        calculated_roe = net_margin * asset_turnover * equity_multiplier
        direct_roe = net_income / equity if equity else 0
        
        dupont_data.append({
            'date': date,
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'calculated_roe': calculated_roe,
            'direct_roe': direct_roe
        })
    
    # 計算趨勢變化（如果有足夠數據）
    trends = []
    if len(dupont_data) >= 2:
        current = dupont_data[0]
        previous = dupont_data[1]
        
        trends.append({
            'factor': '淨利率變化',
            'change': current['net_margin'] - previous['net_margin']
        })
        trends.append({
            'factor': '資產周轉率變化',
            'change': current['asset_turnover'] - previous['asset_turnover']
        })
        trends.append({
            'factor': '權益乘數變化',
            'change': current['equity_multiplier'] - previous['equity_multiplier']
        })
        trends.append({
            'factor': 'ROE變化',
            'change': current['direct_roe'] - previous['direct_roe']
        })
    
    return {
        'annual_data': dupont_data,
        'trends': trends
    }

# 函數：計算現金流分析
def calculate_cashflow_analysis(financial_data):
    """計算現金流分析"""
    
    if not financial_data:
        return None
    
    current = financial_data[0]
    
    # 關鍵指標計算
    operating_cf = current.get('netcashprovidedbyusedinoperatingactivities', 0)
    net_income = current.get('netincomeloss', 1)
    capex = current.get('paymentstoacquireproductiveassets', 0)
    
    # 營運現金流品質比率
    cf_quality_ratio = operating_cf / net_income if net_income else 0
    
    # 自由現金流
    free_cashflow = operating_cf - capex
    
    # 品質評估
    if cf_quality_ratio >= 1.2:
        quality_assessment = "優秀"
        quality_emoji = "😊"
    elif cf_quality_ratio >= 1.0:
        quality_assessment = "良好"
        quality_emoji = "🙂"
    elif cf_quality_ratio >= 0.8:
        quality_assessment = "尚可"
        quality_emoji = "😐"
    else:
        quality_assessment = "需關注"
        quality_emoji = "😰"
    
    # 現金流結構分析
    investing_cf = current.get('netcashprovidedbyusedininvestingactivities', 0)
    financing_cf = current.get('netcashprovidedbyusedinfinancingactivities', 0)
    
    structure_analysis = [
        {'type': '營運現金流', 'amount': operating_cf},
        {'type': '投資現金流', 'amount': investing_cf},
        {'type': '融資現金流', 'amount': financing_cf}
    ]
    
    # 詳細現金流數據
    detailed_data = {
        'operating_cf': operating_cf,
        'investing_cf': investing_cf,
        'financing_cf': financing_cf,
        'net_income': net_income,
        'capex': capex,
        'total_cf': operating_cf + investing_cf + financing_cf
    }
    
    return {
        'cf_quality_ratio': cf_quality_ratio,
        'free_cashflow': free_cashflow,
        'quality_assessment': quality_assessment,
        'quality_emoji': quality_emoji,
        'structure_analysis': structure_analysis,
        'detailed_data': detailed_data
    }

# 函數：處理財務數據用於展示
def process_financial_data_for_display(financial_data):
    """將財務數據轉換為適合展示的DataFrame格式"""
    
    try:
        income_data = []
        balance_data = []
        cash_data = []
        
        for year_data in financial_data:
            date = year_data.get('date', '')
            
            # 損益表數據
            income_row = {
                'Date': date,
                'Revenue': year_data.get('revenues', 0),
                'Gross Profit': year_data.get('grossprofit', 0),
                'Operating Income': year_data.get('operatingincomeloss', 0),
                'Net Income': year_data.get('netincomeloss', 0)
            }
            income_data.append(income_row)
            
            # 資產負債表數據
            balance_row = {
                'Date': date,
                'Total Assets': year_data.get('assets', 0),
                'Current Assets': year_data.get('assetscurrent', 0),
                'Total Liabilities': year_data.get('liabilities', 0),
                'Current Liabilities': year_data.get('liabilitiescurrent', 0),
                'Stockholders Equity': year_data.get('stockholdersequity', 0)
            }
            balance_data.append(balance_row)
            
            # 現金流量表數據
            cash_row = {
                'Date': date,
                'Operating Cash Flow': year_data.get('netcashprovidedbyusedinoperatingactivities', 0),
                'Investing Cash Flow': year_data.get('netcashprovidedbyusedininvestingactivities', 0),
                'Financing Cash Flow': year_data.get('netcashprovidedbyusedinfinancingactivities', 0),
                'Capital Expenditures': year_data.get('paymentstoacquireproductiveassets', 0)
            }
            cash_data.append(cash_row)
        
        # 轉換為DataFrame並設置索引
        income_df = pd.DataFrame(income_data).set_index('Date')
        balance_df = pd.DataFrame(balance_data).set_index('Date')
        cash_df = pd.DataFrame(cash_data).set_index('Date')
        
        # 反轉順序以使最新數據在前
        income_df = income_df.iloc[::-1]
        balance_df = balance_df.iloc[::-1]
        cash_df = cash_df.iloc[::-1]
        
        return income_df, balance_df, cash_df
        
    except Exception as e:
        raise Exception(f"處理財務數據時發生錯誤: {str(e)}")

# 函數：AI分析
def analyze_with_openai(financial_data, fscore_result, zscore_result, dupont_result, cashflow_result, quality_report, stock_info, openai_api_key):
    """使用OpenAI進行台股財務分析"""
    
    if not openai_api_key:
        return "請提供OpenAI API金鑰以進行AI分析"
    
    try:
        # 初始化OpenAI客戶端
        client = OpenAI(api_key=openai_api_key)
        
        # 準備分析數據
        analysis_data = {
            "公司基本資訊": stock_info,
            "財務數據品質報告": quality_report,
            "Piotroski F-Score結果": fscore_result,
            "Altman Z-Score結果": zscore_result,
            "杜邦分析結果": dupont_result,
            "現金流分析結果": cashflow_result,
            "最新年度財務數據": financial_data[0] if financial_data else {}
        }
        
        # 系統訊息 - 設定AI角色
        system_message = """你是一位專精台股財務分析和台灣會計準則的專業分析師，具備以下專業能力：

1. 深度理解台股市場特性和台灣企業經營環境
2. 熟悉台灣會計準則(TIFRS)與國際準則的差異
3. 精通四階段財務分析方法的應用和解讀
4. 了解FinMind開源資料的特點和限制性
5. 擅長客觀分析和教育性解說

你的分析目標：
- 基於已計算完成的四階段分析結果進行專業解讀
- 提供客觀、教育性的財務健康診斷
- 強調台股市場特性和投資環境考量
- 說明資料來源限制和計算欄位的影響
- 避免提供投資建議，專注於教育性分析"""

        # 用戶提示語
        user_prompt = f"""請對以下台股公司進行專業財務分析：

## 分析數據
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

## 分析要求
請進行四階段財務分析解讀，並提供以下結構化分析：

### 1. 資料來源與限制說明
- 說明FinMind開源資料特點
- 標註計算欄位來源和限制
- 解釋對分析結果的影響

### 2. Piotroski F-Score 深度解讀
- 分析9項指標的投資意義
- 解讀各類別得分狀況
- 結合台股市場特性分析

### 3. Altman Z-Score 風險評估
- 解讀風險等級判斷
- 分析各組成要素影響
- 考慮台灣企業特性

### 4. 杜邦分析趨勢洞察
- 分析ROE三因子變化趨勢
- 識別主要驅動因子
- 發現財務效率變化

### 5. 現金流結構分析
- 評估現金流品質
- 分析資本支出模式
- 檢視獲利品質一致性

### 6. 綜合財務健康診斷
請輸出四階段評分總結表格：
| 分析階段 | 評分狀態 | 評價 | 主要發現 |

### 7. 台股投資環境考量
- 產業政策影響
- 兩岸關係考量
- 法規環境分析

### 8. 分析結論
- **主要優勢**：3-5個關鍵優勢
- **風險因素**：需關注的風險點
- **後續追蹤重點**：監控的關鍵指標

請確保分析客觀專業，強調教育用途，避免投資建議。"""

        # 調用OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI分析時發生錯誤：{str(e)}"

# 函數：創建圖表
def create_financial_charts(income_df, balance_df, cash_df):
    """創建專業財務圖表"""
    
    charts = {}
    
    # 損益表柱狀圖
    fig_income = go.Figure()
    for column in income_df.columns:
        fig_income.add_trace(go.Bar(
            x=income_df.index,
            y=income_df[column],
            name=column
        ))
    
    fig_income.update_layout(
        title="損益表關鍵指標趨勢",
        xaxis_title="年度",
        yaxis_title="金額 (千元)",
        barmode='group',
        template='plotly_white',
        height=500
    )
    charts['income'] = fig_income
    
    # 資產負債表圖表
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(
        x=balance_df.index,
        y=balance_df['Total Assets'],
        mode='lines+markers',
        name='總資產',
        line=dict(color='steelblue', width=3)
    ))
    fig_balance.add_trace(go.Scatter(
        x=balance_df.index,
        y=balance_df['Total Liabilities'],
        mode='lines+markers',
        name='總負債',
        line=dict(color='darkred', width=3)
    ))
    fig_balance.add_trace(go.Scatter(
        x=balance_df.index,
        y=balance_df['Stockholders Equity'],
        mode='lines+markers',
        name='股東權益',
        line=dict(color='darkgreen', width=3)
    ))
    
    fig_balance.update_layout(
        title="資產負債表趨勢分析",
        xaxis_title="年度",
        yaxis_title="金額 (千元)",
        template='plotly_white',
        height=500
    )
    charts['balance'] = fig_balance
    
    # 現金流量圖表
    fig_cash = go.Figure()
    fig_cash.add_trace(go.Bar(
        x=cash_df.index,
        y=cash_df['Operating Cash Flow'],
        name='營運現金流',
        marker_color='darkgreen'
    ))
    fig_cash.add_trace(go.Bar(
        x=cash_df.index,
        y=cash_df['Investing Cash Flow'],
        name='投資現金流',
        marker_color='goldenrod'
    ))
    fig_cash.add_trace(go.Bar(
        x=cash_df.index,
        y=cash_df['Financing Cash Flow'],
        name='融資現金流',
        marker_color='purple'
    ))
    
    fig_cash.update_layout(
        title="現金流量分析",
        xaxis_title="年度",
        yaxis_title="金額 (千元)",
        template='plotly_white',
        height=500
    )
    charts['cash'] = fig_cash
    
    return charts

# 函數：創建F-Score圓餅圖
def create_fscore_pie_chart(fscore_result):
    """創建F-Score通過率圓餅圖"""
    
    total_tests = 9
    passed_tests = fscore_result['total_score']
    failed_tests = total_tests - passed_tests
    
    fig = go.Figure(data=[go.Pie(
        labels=['通過', '未通過'],
        values=[passed_tests, failed_tests],
        marker_colors=['green', 'red'],
        hole=0.3
    )])
    
    fig.update_layout(
        title=f"F-Score 通過率 ({passed_tests}/9)",
        template='plotly_white',
        height=400
    )
    
    return fig

# 函數：創建Z-Score儀表盤
def create_zscore_gauge(zscore_result):
    """創建Z-Score風險儀表盤"""
    
    z_score = zscore_result['z_score']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = z_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Altman Z-Score"},
        delta = {'reference': 2.99},
        gauge = {
            'axis': {'range': [None, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1.81], 'color': "red"},
                {'range': [1.81, 2.99], 'color': "yellow"},
                {'range': [2.99, 5], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': z_score
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    return fig

# 主應用程式
def main():
    # 側邊欄
    st.sidebar.header("Code Gym", divider="rainbow")
    
    # 股票代碼輸入
    ticker = st.sidebar.text_input(
        "輸入台股代碼（例如：2330 代表台積電）", 
        "2330",
        help="請輸入四位數字的台股代碼"
    )
    
    # FinMind API Token
    finmind_api_token = st.sidebar.text_input(
        "輸入 FinMind API Token", 
        type="password", 
        value="",
        help="請前往 FinMind 官網申請 API Token"
    )
    
    # OpenAI API金鑰
    openai_api_key = st.sidebar.text_input(
        "輸入OpenAI API金鑰", 
        type="password", 
        value="",
        help="用於AI財務分析功能"
    )
    
    # 起始日期 - 自動設定為當前日期往前推五年
    default_start_date = datetime.now() - timedelta(days=5*365)  # 往前推5年
    start_date = st.sidebar.date_input(
        "數據起始日期",
        value=default_start_date,
        help="選擇財務數據的起始日期（預設為5年前）"
    ).strftime('%Y-%m-%d')
    
    # 免責聲明
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📢 免責聲明
    本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
    請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
    """)
    
    # 執行分析按鈕
    if st.sidebar.button("分析股票", type="primary"):
        # 驗證台股代碼
        is_valid, error_msg = validate_taiwan_stock_code(ticker)
        if not is_valid:
            st.error(error_msg)
            return
        
        if not finmind_api_token:
            st.warning("請輸入 FinMind API Token")
            return
        
        try:
            # 從FinMind API獲取數據
            finmind_data = get_finmind_data_from_apis(ticker, finmind_api_token, start_date)
            
            # 顯示基本資訊
            try:
                stock_info_data = finmind_data.get('stock_info', [])
                key_metrics_data = finmind_data.get('key_metrics', [])
                
                if stock_info_data:
                    stock_info = stock_info_data[0]
                    company_name = stock_info.get('stock_name', ticker)
                    industry = stock_info.get('industry_category', 'N/A')
                else:
                    company_name = ticker
                    industry = 'N/A'
                
                # 獲取最新的關鍵指標
                latest_metrics = key_metrics_data[0] if key_metrics_data else {}
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader(f"{company_name} ({ticker})")
                    st.write(f"**產業類別:** {industry}")
                    st.write(f"**股票代碼:** {ticker}")
                
                with col2:
                    # 計算市值
                    latest_financial = finmind_data['financial_statements'][0] if finmind_data['financial_statements'] else {}
                    market_cap = latest_financial.get('market_capitalization', 0)
                    st.metric("計算市值", format_large_number(market_cap))
                
                with col3:
                    # 顯示股東權益
                    equity = latest_financial.get('stockholdersequity', 0)
                    st.metric("股東權益", format_large_number(equity))
                    
            except Exception as e:
                st.error(f"獲取基本資訊時發生錯誤：{str(e)}")
            
            # 處理財務數據
            try:
                financial_data = finmind_data['financial_statements']
                income_df, balance_df, cash_df = process_financial_data_for_display(financial_data)
                
                if income_df.empty:
                    st.error("無法獲取有效的財務數據")
                    return
                                       
            except Exception as e:
                st.error(f"處理財報資料時發生錯誤：{str(e)}")
                return
            
            # 創建財報分析和AI分析標籤
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "損益表分析", "資產負債表分析", "現金流量表分析", "四階段財報分析", "AI分析"
            ])
            
            with tab1:
                st.subheader("損益表分析")
                
                try:
                    if not income_df.empty:
                        # 創建圖表
                        charts = create_financial_charts(income_df, balance_df, cash_df)
                        
                        # 顯示損益表圖表
                        st.plotly_chart(charts['income'], use_container_width=True)
                        
                        # 顯示完整損益表
                        st.subheader("完整損益表")
                        st.dataframe(income_df, use_container_width=True)
                    else:
                        st.write("沒有可用的損益表資料")
                        
                except Exception as e:
                    st.error(f"顯示損益表時發生錯誤：{str(e)}")
            
            with tab2:
                st.subheader("資產負債表分析")
                
                try:
                    if not balance_df.empty:
                        # 創建圖表
                        charts = create_financial_charts(income_df, balance_df, cash_df)
                        
                        # 顯示資產負債表圖表
                        st.plotly_chart(charts['balance'], use_container_width=True)
                        
                        # 顯示完整資產負債表
                        st.subheader("完整資產負債表")
                        st.dataframe(balance_df, use_container_width=True)
                    else:
                        st.write("沒有可用的資產負債表資料")
                        
                except Exception as e:
                    st.error(f"顯示資產負債表時發生錯誤：{str(e)}")
            
            with tab3:
                st.subheader("現金流量表分析")
                
                try:
                    if not cash_df.empty:
                        # 創建圖表
                        charts = create_financial_charts(income_df, balance_df, cash_df)
                        
                        # 顯示現金流量圖表
                        st.plotly_chart(charts['cash'], use_container_width=True)
                        
                        # 顯示完整現金流量表
                        st.subheader("完整現金流量表")
                        st.dataframe(cash_df, use_container_width=True)
                    else:
                        st.write("沒有可用的現金流量表資料")
                        
                except Exception as e:
                    st.error(f"顯示現金流量表時發生錯誤：{str(e)}")
            
            with tab4:
                st.subheader("四階段財報分析")
                
                try:
                    # 數據品質檢查
                    quality_report = analyze_data_quality(financial_data, finmind_data.get('raw_finmind_data', {}))
                    
                    # 顯示數據品質報告
                    with st.expander("📊 數據品質報告", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("數據完整性", quality_report["數據完整性"])
                            st.metric("數據年份", f"{quality_report['數據年份']} 年")
                        
                        with col2:
                            if quality_report["缺失欄位"]:
                                st.warning("⚠️ 缺失欄位：")
                                for field in quality_report["缺失欄位"]:
                                    st.write(f"- {field}")
                        
                        if quality_report["計算欄位說明"]:
                            st.info("ℹ️ 計算欄位說明：")
                            for explanation in quality_report["計算欄位說明"]:
                                st.write(f"- {explanation}")
                    
                    # 階段一：Piotroski F-Score
                    st.markdown("### 📈 階段一：Piotroski F-Score")
                    fscore_result = calculate_piotroski_fscore(financial_data)
                    
                    if fscore_result:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # F-Score總分
                            total_score = fscore_result['total_score']
                            st.metric("F-Score 總分", f"{total_score}/9", help="Piotroski F-Score 總分")
                            
                            # 獲利能力指標表格
                            st.markdown("**獲利能力指標 (4項)**")
                            prof_df = pd.DataFrame(fscore_result['profitability_scores'])
                            st.dataframe(prof_df, use_container_width=True, hide_index=True)
                            
                            # 槓桿與流動性指標表格
                            st.markdown("**槓桿與流動性指標 (3項)**")
                            lev_df = pd.DataFrame(fscore_result['leverage_scores'])
                            st.dataframe(lev_df, use_container_width=True, hide_index=True)
                            
                            # 營運效率指標表格
                            st.markdown("**營運效率指標 (2項)**")
                            eff_df = pd.DataFrame(fscore_result['efficiency_scores'])
                            st.dataframe(eff_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # F-Score圓餅圖
                            fig_pie = create_fscore_pie_chart(fscore_result)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    else:
                        st.warning("財務數據不足，無法計算 Piotroski F-Score（需要至少2年數據）")
                    
                    st.markdown("---")
                    
                    # 階段二：Altman Z-Score
                    st.markdown("### ⚖️ 階段二：Altman Z-Score")
                    zscore_result = calculate_altman_zscore(financial_data)
                    
                    if zscore_result:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Z-Score總分和風險等級
                            z_score = zscore_result['z_score']
                            risk_level = zscore_result['risk_level']
                            risk_emoji = zscore_result['risk_emoji']
                            
                            st.metric("Altman Z-Score", f"{z_score:.2f}")
                            st.metric("風險等級", f"{risk_level} {risk_emoji}")
                            
                            # Z-Score組成要素表格
                            st.markdown("**Z-Score 組成要素**")
                            components_data = []
                            for key, comp in zscore_result['components'].items():
                                components_data.append({
                                    '項目': f"{key}項",
                                    '描述': comp['description'],
                                    '比率值': f"{comp['ratio']:.4f}",
                                    '權重後數值': f"{comp['weighted']:.4f}"
                                })
                            comp_df = pd.DataFrame(components_data)
                            st.dataframe(comp_df, use_container_width=True, hide_index=True)
                            
                            # 計算基礎數據表格
                            st.markdown("**計算基礎數據**")
                            base_data = zscore_result['base_data']
                            base_df = pd.DataFrame([
                                {'項目': '營運資本', '金額': format_large_number(base_data['working_capital'])},
                                {'項目': '總資產', '金額': format_large_number(base_data['total_assets'])},
                                {'項目': '保留盈餘', '金額': format_large_number(base_data['retained_earnings'])},
                                {'項目': 'EBIT', '金額': format_large_number(base_data['ebit'])},
                                {'項目': '市值', '金額': format_large_number(base_data['market_cap'])},
                                {'項目': '總負債', '金額': format_large_number(base_data['total_liabilities'])},
                                {'項目': '營收', '金額': format_large_number(base_data['revenues'])}
                            ])
                            st.dataframe(base_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Z-Score儀表盤
                            fig_gauge = create_zscore_gauge(zscore_result)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    else:
                        st.warning("無法計算 Altman Z-Score")
                    
                    st.markdown("---")
                    
                    # 階段三：杜邦分析
                    st.markdown("### 🔍 階段三：杜邦分析")
                    dupont_result = calculate_dupont_analysis(financial_data)
                    
                    if dupont_result['annual_data']:
                        # 當前ROE指標
                        current_roe = dupont_result['annual_data'][0]['direct_roe']
                        st.metric("當前 ROE", f"{current_roe:.2%}")
                        
                        # 年度杜邦分析表格
                        st.markdown("**年度杜邦分析**")
                        dupont_df = pd.DataFrame([
                            {
                                '日期': item['date'],
                                '淨利率': f"{item['net_margin']:.4f}",
                                '資產周轉率': f"{item['asset_turnover']:.4f}",
                                '權益乘數': f"{item['equity_multiplier']:.4f}",
                                '計算ROE': f"{item['calculated_roe']:.4f}",
                                '直接ROE': f"{item['direct_roe']:.4f}"
                            }
                            for item in dupont_result['annual_data']
                        ])
                        st.dataframe(dupont_df, use_container_width=True, hide_index=True)
                        
                        # 趨勢變化分析表格
                        if dupont_result['trends']:
                            st.markdown("**趨勢變化分析**")
                            trends_df = pd.DataFrame([
                                {
                                    '因子': item['factor'],
                                    '變化': f"{item['change']:.4f}"
                                }
                                for item in dupont_result['trends']
                            ])
                            st.dataframe(trends_df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.warning("無法計算杜邦分析")
                    
                    st.markdown("---")
                    
                    # 階段四：現金流分析
                    st.markdown("### 💰 階段四：現金流分析")
                    cashflow_result = calculate_cashflow_analysis(financial_data)
                    
                    if cashflow_result:
                        # 現金流品質指標
                        cf_quality_ratio = cashflow_result['cf_quality_ratio']
                        quality_assessment = cashflow_result['quality_assessment']
                        quality_emoji = cashflow_result['quality_emoji']
                        
                        st.metric(
                            "現金流品質比率", 
                            f"{cf_quality_ratio:.2f}",
                            delta=f"{quality_assessment} {quality_emoji}"
                        )
                        
                        # 現金流關鍵指標表格
                        st.markdown("**現金流關鍵指標**")
                        cf_key_df = pd.DataFrame([
                            {
                                '指標': '營運現金流品質比率',
                                '數值': f"{cf_quality_ratio:.2f}",
                                '評估': quality_assessment
                            },
                            {
                                '指標': '自由現金流',
                                '數值': format_large_number(cashflow_result['free_cashflow']),
                                '評估': '正值為佳' if cashflow_result['free_cashflow'] > 0 else '需關注'
                            }
                        ])
                        st.dataframe(cf_key_df, use_container_width=True, hide_index=True)
                        
                        # 現金流結構分析表格
                        st.markdown("**現金流結構分析**")
                        structure_df = pd.DataFrame([
                            {
                                '類型': item['type'],
                                '金額': format_large_number(item['amount'])
                            }
                            for item in cashflow_result['structure_analysis']
                        ])
                        st.dataframe(structure_df, use_container_width=True, hide_index=True)
                        
                        # 詳細現金流數據表格
                        st.markdown("**詳細現金流數據**")
                        detailed_data = cashflow_result['detailed_data']
                        detailed_df = pd.DataFrame([
                            {'項目': '營運現金流', '金額': format_large_number(detailed_data['operating_cf'])},
                            {'項目': '投資現金流', '金額': format_large_number(detailed_data['investing_cf'])},
                            {'項目': '融資現金流', '金額': format_large_number(detailed_data['financing_cf'])},
                            {'項目': '淨利潤', '金額': format_large_number(detailed_data['net_income'])},
                            {'項目': '資本支出', '金額': format_large_number(detailed_data['capex'])},
                            {'項目': '現金流總計', '金額': format_large_number(detailed_data['total_cf'])}
                        ])
                        st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.warning("無法計算現金流分析")
                        
                except Exception as e:
                    st.error(f"四階段財報分析時發生錯誤：{str(e)}")
            
            with tab5:
                st.subheader("AI 綜合財務分析")
                
                # 檢查是否已完成四階段分析
                try:
                    # 重新計算所有分析結果用於AI分析
                    quality_report = analyze_data_quality(financial_data, finmind_data.get('raw_finmind_data', {}))
                    fscore_result = calculate_piotroski_fscore(financial_data)
                    zscore_result = calculate_altman_zscore(financial_data)
                    dupont_result = calculate_dupont_analysis(financial_data)
                    cashflow_result = calculate_cashflow_analysis(financial_data)
                    
                    # 準備公司資訊
                    stock_info = {
                        'company_name': company_name,
                        'stock_code': ticker,
                        'industry': industry
                    }
                    
                    # 自動進行AI分析
                    if not openai_api_key:
                        st.warning("請在側邊欄輸入 OpenAI API 金鑰以使用 AI 分析功能")
                        st.info("💡 AI分析功能需要OpenAI API金鑰，請在左側邊欄輸入後重新分析")
                    else:
                        with st.spinner("正在使用AI進行四階段財務分析..."):
                            ai_analysis = analyze_with_openai(
                                financial_data,
                                fscore_result,
                                zscore_result,
                                dupont_result,
                                cashflow_result,
                                quality_report,
                                stock_info,
                                openai_api_key
                            )
                            
                            st.markdown("### 🎯 AI 財務分析報告")
                            st.markdown(ai_analysis)
                    
                    # 顯示分析摘要
                    st.markdown("### 📋 分析數據摘要")
                    
                    summary_data = []
                    
                    if fscore_result:
                        summary_data.append({
                            '分析項目': 'Piotroski F-Score',
                            '結果': f"{fscore_result['total_score']}/9",
                            '狀態': '優秀' if fscore_result['total_score'] >= 7 else '良好' if fscore_result['total_score'] >= 5 else '需關注'
                        })
                    
                    if zscore_result:
                        summary_data.append({
                            '分析項目': 'Altman Z-Score',
                            '結果': f"{zscore_result['z_score']:.2f}",
                            '狀態': zscore_result['risk_level']
                        })
                    
                    if dupont_result['annual_data']:
                        current_roe = dupont_result['annual_data'][0]['direct_roe']
                        summary_data.append({
                            '分析項目': 'ROE (杜邦分析)',
                            '結果': f"{current_roe:.2%}",
                            '狀態': '優秀' if current_roe > 0.15 else '良好' if current_roe > 0.10 else '需關注'
                        })
                    
                    if cashflow_result:
                        summary_data.append({
                            '分析項目': '現金流品質',
                            '結果': f"{cashflow_result['cf_quality_ratio']:.2f}",
                            '狀態': cashflow_result['quality_assessment']
                        })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"準備AI分析時發生錯誤：{str(e)}")
                    
        except Exception as e:
            st.error(f"分析過程中發生錯誤：{str(e)}")
            
            # 顯示詳細錯誤信息用於調試
            with st.expander("詳細錯誤信息", expanded=False):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()