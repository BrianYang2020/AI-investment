"""
台股財報分析報告 PDF 產生模組

使用 reportlab 產生 PDF（含中文字型、圖表、表格與 AI 分析報告）。
中文字型依序尋找：tw_financial/fonts/ 內的字型檔 → 系統字型 → reportlab 內建 CID 字型。
"""

import os
import re
import glob
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 系統中常見的中文 TrueType 字型（需為 TrueType 外框，reportlab 不支援 CFF 格式的 OTF）
SYSTEM_FONT_CANDIDATES = [
    ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 0),       # Linux：文泉驛正黑（Streamlit Cloud 以 packages.txt 安裝）
    ('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 0),     # Linux：文泉驛微米黑
    ('/usr/share/fonts/truetype/arphic/uming.ttc', 0),         # Linux：AR PL UMing
    ('C:/Windows/Fonts/msjh.ttc', 0),                          # Windows：微軟正黑體
    ('C:/Windows/Fonts/mingliu.ttc', 0),                       # Windows：細明體
    ('/Library/Fonts/Arial Unicode.ttf', 0),                   # macOS
]

FONT_NAME = 'ReportCJK'
CHART_WIDTH = 510
PAGE_MARGIN = 15 * mm

# 專業配色
COLOR_STEELBLUE = colors.HexColor('#4682B4')
COLOR_DARKGREEN = colors.HexColor('#006400')
COLOR_GOLDENROD = colors.HexColor('#DAA520')
COLOR_DARKRED = colors.HexColor('#8B0000')
COLOR_PURPLE = colors.HexColor('#800080')
COLOR_HEADER_BG = colors.HexColor('#2F4F4F')
COLOR_ROW_ALT = colors.HexColor('#F2F5F7')

_registered_font = None

def register_chinese_font():
    """註冊中文字型，回傳可用的字型名稱（只註冊一次）"""
    global _registered_font
    if _registered_font:
        return _registered_font

    # 1. 專案內附字型（可自行放入 tw_financial/fonts/*.ttf 或 *.ttc）
    bundled = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', '*.tt[fc]')))
    candidates = [(path, 0) for path in bundled] + SYSTEM_FONT_CANDIDATES

    for path, subfont_index in candidates:
        if not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, path, subfontIndex=subfont_index))
            pdfmetrics.registerFontFamily(FONT_NAME, normal=FONT_NAME, bold=FONT_NAME,
                                          italic=FONT_NAME, boldItalic=FONT_NAME)
            _registered_font = FONT_NAME
            return _registered_font
        except Exception:
            continue

    # 2. 找不到字型檔時，使用 reportlab 內建的繁體中文 CID 字型（不內嵌，由 PDF 閱讀器提供字型）
    cid_font = 'MSung-Light'
    pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
    pdfmetrics.registerFontFamily(cid_font, normal=cid_font, bold=cid_font, italic=cid_font, boldItalic=cid_font)
    _registered_font = cid_font
    return _registered_font

# =====================================================================
# 樣式與工具
# =====================================================================

def build_styles(font):
    """建立段落樣式"""
    base = dict(fontName=font, leading=15, fontSize=10, wordWrap='CJK')
    return {
        'title': ParagraphStyle('title', **{**base, 'fontSize': 18, 'leading': 24, 'textColor': COLOR_HEADER_BG, 'spaceAfter': 4}),
        'subtitle': ParagraphStyle('subtitle', **{**base, 'fontSize': 9, 'textColor': colors.grey, 'spaceAfter': 8}),
        'h1': ParagraphStyle('h1', **{**base, 'fontSize': 14, 'leading': 20, 'textColor': COLOR_HEADER_BG, 'spaceBefore': 12, 'spaceAfter': 6}),
        'h2': ParagraphStyle('h2', **{**base, 'fontSize': 12, 'leading': 17, 'textColor': COLOR_STEELBLUE, 'spaceBefore': 8, 'spaceAfter': 4}),
        'h3': ParagraphStyle('h3', **{**base, 'fontSize': 10.5, 'leading': 15, 'textColor': COLOR_STEELBLUE, 'spaceBefore': 6, 'spaceAfter': 2}),
        'body': ParagraphStyle('body', **base),
        'bullet': ParagraphStyle('bullet', **{**base, 'leftIndent': 12, 'bulletIndent': 2}),
        'small': ParagraphStyle('small', **{**base, 'fontSize': 8, 'leading': 11, 'textColor': colors.grey}),
        'cell': ParagraphStyle('cell', **{**base, 'fontSize': 8.5, 'leading': 11}),
        'cell_header': ParagraphStyle('cell_header', **{**base, 'fontSize': 8.5, 'leading': 11, 'textColor': colors.white}),
    }

def escape_text(text):
    """跳脫 reportlab 段落標記字元"""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

# 中文字型通常不含勾叉符號與 emoji，轉換為字型內有的字元，避免 PDF 出現空白
SYMBOL_REPLACEMENTS = {'✓': '○', '✔': '○', '✅': '○', '✗': '×', '✘': '×', '❌': '×', '⚠️': '注意：', '⚠': '注意：'}
EMOJI_PATTERN = re.compile('[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]')

def sanitize_symbols(text):
    """替換字型不支援的符號並移除 emoji"""
    for symbol, replacement in SYMBOL_REPLACEMENTS.items():
        text = text.replace(symbol, replacement)
    return EMOJI_PATTERN.sub('', text)

def inline_markdown(text):
    """處理行內 Markdown：粗體、行內程式碼"""
    text = escape_text(sanitize_symbols(str(text)))
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text

def make_table(rows, styles, col_widths=None, header=True):
    """建立表格（自動換行、表頭底色、斑馬紋）"""
    if not rows:
        return Spacer(1, 0)
    data = []
    for r, row in enumerate(rows):
        style = styles['cell_header'] if header and r == 0 else styles['cell']
        data.append([Paragraph(inline_markdown('' if cell is None else cell), style) for cell in row])

    n_cols = max(len(row) for row in data)
    data = [row + [''] * (n_cols - len(row)) for row in data]
    if col_widths is None:
        col_widths = [CHART_WIDTH / n_cols] * n_cols

    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ('GRID', (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(('BACKGROUND', (0, 0), (-1, 0), COLOR_HEADER_BG))
    for r in range(1 if header else 0, len(data)):
        if r % 2 == 0:
            commands.append(('BACKGROUND', (0, r), (-1, r), COLOR_ROW_ALT))
    table.setStyle(TableStyle(commands))
    return table

def dataframe_rows(df, max_rows=8, index_name='季度'):
    """將 DataFrame（日期升冪）轉為表格列：最新在前、最多 max_rows 筆"""
    recent = df.iloc[::-1].head(max_rows)
    rows = [[index_name] + list(recent.columns)]
    for idx, row in recent.iterrows():
        rows.append([str(idx)] + ['N/A' if value != value else f"{value:,.2f}" for value in row.values])
    return rows

def clean_series(values):
    """圖表數值：NaN 轉為 None（reportlab 視為缺值）"""
    return [None if value != value else float(value) for value in values]

# =====================================================================
# 圖表
# =====================================================================

def add_chart_title_and_legend(drawing, title, names, color_list, font, height):
    drawing.add(String(CHART_WIDTH / 2, height - 12, title, fontName=font, fontSize=10, textAnchor='middle'))
    legend = Legend()
    legend.x = CHART_WIDTH - 10
    legend.y = height - 22
    legend.alignment = 'right'
    legend.boxAnchor = 'ne'
    legend.columnMaximum = 1
    legend.deltax = 70
    legend.dx = 8
    legend.dy = 8
    legend.fontName = font
    legend.fontSize = 8
    legend.colorNamePairs = list(zip(color_list, names))
    drawing.add(legend)

def configure_axes(chart, font):
    chart.categoryAxis.labels.fontName = font
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.valueAxis.labels.fontName = font
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = lambda v: f"{v:,.0f}"
    chart.valueAxis.forceZero = 1                # 數值軸從 0 開始，避免柱狀高度誤導
    chart.categoryAxis.labelAxisMode = 'low'     # 季度標籤固定在圖表底部（含負值時不與柱狀重疊）

def bar_chart(title, df, columns, color_list, font, height=210):
    """群組柱狀圖（單位：億元）"""
    drawing = Drawing(CHART_WIDTH, height)
    chart = VerticalBarChart()
    chart.x, chart.y = 50, 40
    chart.width, chart.height = CHART_WIDTH - 65, height - 80
    chart.data = [clean_series(df[col].values) for col in columns]
    chart.categoryAxis.categoryNames = [str(i) for i in df.index]
    chart.groupSpacing = 6
    chart.barSpacing = 1
    for i, color in enumerate(color_list):
        chart.bars[i].fillColor = color
        chart.bars[i].strokeColor = None
    configure_axes(chart, font)
    drawing.add(chart)
    add_chart_title_and_legend(drawing, title, columns, color_list, font, height)
    return drawing

def line_chart(title, df, columns, color_list, font, height=210):
    """折線圖（單位：億元）"""
    drawing = Drawing(CHART_WIDTH, height)
    chart = HorizontalLineChart()
    chart.x, chart.y = 50, 40
    chart.width, chart.height = CHART_WIDTH - 65, height - 80
    chart.data = [clean_series(df[col].values) for col in columns]
    chart.categoryAxis.categoryNames = [str(i) for i in df.index]
    chart.joinedLines = 1
    for i, color in enumerate(color_list):
        chart.lines[i].strokeColor = color
        chart.lines[i].strokeWidth = 2
    configure_axes(chart, font)
    drawing.add(chart)
    add_chart_title_and_legend(drawing, title, columns, color_list, font, height)
    return drawing

# =====================================================================
# Markdown（AI 報告）轉 PDF 段落
# =====================================================================

def markdown_to_flowables(markdown_text, styles):
    """將 AI 回傳的 Markdown 轉為 PDF 段落（支援標題、清單、表格、粗體）"""
    flowables = []
    table_lines = []

    def flush_table():
        if not table_lines:
            return
        rows = []
        for line in table_lines:
            # 略過分隔列，例如 |---|:---:|
            if re.match(r'^\|?\s*:?-{2,}', line.replace(' ', '')):
                continue
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            rows.append(cells)
        if rows:
            flowables.append(make_table(rows, styles))
            flowables.append(Spacer(1, 6))
        table_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith('|'):
            table_lines.append(stripped)
            continue
        flush_table()

        if not stripped:
            flowables.append(Spacer(1, 4))
            continue
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', stripped):
            flowables.append(Spacer(1, 6))
            continue

        heading = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if heading:
            level = len(heading.group(1))
            style = styles['h2'] if level <= 2 else styles['h3']
            flowables.append(Paragraph(inline_markdown(heading.group(2)), style))
            continue

        indent_level = (len(line) - len(line.lstrip())) // 2
        bullet = re.match(r'^[-*•]\s+(.*)$', stripped)
        numbered = re.match(r'^(\d+[.)])\s+(.*)$', stripped)
        if bullet or numbered:
            style = ParagraphStyle('bullet_level', parent=styles['bullet'],
                                   leftIndent=12 + indent_level * 12, bulletIndent=2 + indent_level * 12)
            if bullet:
                flowables.append(Paragraph(inline_markdown(bullet.group(1)), style, bulletText='•'))
            else:
                flowables.append(Paragraph(inline_markdown(numbered.group(2)), style, bulletText=numbered.group(1)))
            continue

        flowables.append(Paragraph(inline_markdown(stripped), styles['body']))

    flush_table()
    return flowables

# =====================================================================
# 報告主體
# =====================================================================

def generate_pdf_report(report):
    """
    產生財報分析 PDF

    Args:
        report: dict，包含 ticker、company_name、industries、market、basic_info、
                summary_rows、quality_report、fscore_result、zscore_result、dupont_result、
                cashflow_result、income_df、balance_df、ratio_df、cash_df、ai_analysis、generated_at

    Returns:
        bytes: PDF 檔案內容
    """
    font = register_chinese_font()
    styles = build_styles(font)
    buffer = BytesIO()
    generated_at = report.get('generated_at') or datetime.now().strftime('%Y-%m-%d %H:%M')
    title = f"{report['company_name']}（{report['ticker']}）財報分析報告"

    def draw_page_frame(canvas, doc):
        """頁首與頁尾：報告名稱、頁碼、免責提醒"""
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(PAGE_MARGIN, A4[1] - 10 * mm, f"【Code Gym】AI 分析台股基本面應用｜{title}")
        canvas.drawString(PAGE_MARGIN, 8 * mm, "本報告僅供學術研究與教育用途，不構成投資建議或財務建議。")
        canvas.drawRightString(A4[0] - PAGE_MARGIN, 8 * mm, f"第 {doc.page} 頁")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN + 3 * mm, bottomMargin=PAGE_MARGIN,
        title=title, author="Code Gym AI 分析台股基本面應用",
    )

    quality = report['quality_report']
    story = [
        Paragraph(escape_text(title), styles['title']),
        Paragraph(escape_text(
            f"產業：{'、'.join(report.get('industries') or []) or 'N/A'}　｜　市場：{report.get('market', 'N/A')}　｜　"
            f"最新財報：{quality.get('最新財報季度', 'N/A')}　｜　產生時間：{generated_at}"
        ), styles['subtitle']),
    ]

    # 一、基本資訊與分析摘要
    basic = report.get('basic_info', {})
    story.append(Paragraph("一、基本資訊與分析摘要", styles['h1']))
    story.append(make_table([
        ['項目', '數值', '說明'],
        ['最新收盤價', basic.get('close_text', 'N/A'), basic.get('price_date_text', '')],
        ['估算市值', basic.get('market_cap_text', 'N/A'), 'PBR × 歸屬母公司權益'],
        ['本益比 (PER)', basic.get('per_text', 'N/A'), basic.get('metrics_date_text', '')],
    ], styles, col_widths=[110, 150, 250]))
    story.append(Spacer(1, 8))
    summary_rows = report.get('summary_rows') or []
    if summary_rows:
        story.append(make_table([['分析項目', '結果', '狀態']] +
                                [[row['分析項目'], row['結果'], row['狀態']] for row in summary_rows],
                                styles, col_widths=[200, 150, 160]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(escape_text(f"分析基準：{quality.get('分析基準', '')}"), styles['small']))

    # 二、財報趨勢圖
    story.append(Paragraph("二、財報趨勢（單位：億元）", styles['h1']))
    income_df = report['income_df'].tail(12)
    balance_df = report['balance_df'].tail(12)
    cash_df = report['cash_df'].tail(12)
    story.append(bar_chart("損益表關鍵指標（單季）", income_df, ['營收', '毛利', '營業利益', '稅後淨利'],
                           [COLOR_STEELBLUE, COLOR_DARKGREEN, COLOR_GOLDENROD, COLOR_DARKRED], font))
    story.append(line_chart("資產負債表趨勢（季末）", balance_df, ['總資產', '總負債', '股東權益'],
                            [COLOR_STEELBLUE, COLOR_DARKRED, COLOR_DARKGREEN], font))
    story.append(bar_chart("現金流量（單季）", cash_df, ['營運現金流', '投資現金流', '融資現金流', '自由現金流'],
                           [COLOR_DARKGREEN, COLOR_GOLDENROD, COLOR_PURPLE, COLOR_STEELBLUE], font))

    # 三、四階段財報分析
    story.append(Paragraph("三、四階段財報分析", styles['h1']))

    fscore = report.get('fscore_result')
    story.append(Paragraph("階段一：Piotroski F-Score", styles['h2']))
    if fscore:
        story.append(Paragraph(escape_text(
            f"總分 {fscore['total_score']}/9（可判斷 {fscore['evaluated_count']} 項）｜比較基準："
            f"{fscore['current_date']} vs {fscore['previous_date']}"), styles['body']))
        rows = [['類別', '指標', '本期', '前期／比較', '狀態']]
        for category, key in [('獲利能力', 'profitability_scores'), ('槓桿與流動性', 'leverage_scores'), ('營運效率', 'efficiency_scores')]:
            for item in fscore[key]:
                status = {'✓': '通過', '✗': '未通過'}.get(item['狀態'], '資料不足')
                rows.append([category, item['指標'], item['本期'], item['前期／比較'], status])
        story.append(make_table(rows, styles, col_widths=[65, 175, 90, 90, 90]))
    else:
        story.append(Paragraph("財務數據不足，無法計算（需要至少 8 季數據）", styles['body']))

    zscore = report.get('zscore_result')
    story.append(Paragraph("階段二：Altman Z-Score", styles['h2']))
    if zscore:
        z_text = 'N/A' if zscore['z_score'] is None else f"{zscore['z_score']:.2f}"
        story.append(Paragraph(escape_text(
            f"Z-Score：{z_text}｜風險等級：{zscore['risk_level']}｜計算基準：{zscore['date']}"), styles['body']))
        if zscore.get('missing_components'):
            story.append(Paragraph(escape_text(f"缺少資料無法計算：{'、'.join(zscore['missing_components'])}"), styles['body']))
        rows = [['項目', '描述', '比率值', '權重後數值']]
        for key, comp in zscore['components'].items():
            rows.append([f"{key}項", comp['description'],
                         'N/A' if comp['ratio'] is None else f"{comp['ratio']:.4f}",
                         'N/A' if comp['weighted'] is None else f"{comp['weighted']:.4f}"])
        story.append(make_table(rows, styles, col_widths=[60, 200, 125, 125]))

    dupont = report.get('dupont_result') or {}
    story.append(Paragraph("階段三：杜邦分析", styles['h2']))
    if dupont.get('annual_data'):
        def fmt(v, pct=False):
            return 'N/A' if v is None else (f"{v:.2%}" if pct else f"{v:.4f}")
        rows = [['期間', '淨利率', '資產周轉率', '權益乘數', 'ROE']]
        for item in dupont['annual_data']:
            rows.append([item['date'], fmt(item['net_margin'], True), fmt(item['asset_turnover']),
                         fmt(item['equity_multiplier']), fmt(item['direct_roe'], True)])
        story.append(make_table(rows, styles, col_widths=[130, 95, 95, 95, 95]))

    cashflow = report.get('cashflow_result')
    cashflow_heading = Paragraph("階段四：現金流分析", styles['h2'])
    if cashflow:
        rows = [['指標', '數值', '評估']]
        ratio = cashflow['cf_quality_ratio']
        rows.append(['營運現金流品質比率', 'N/A' if ratio is None else f"{ratio:.2f}", cashflow['quality_assessment']])
        for label, value in report.get('cashflow_detail_rows', []):
            rows.append([label, value, ''])
        # 標題與表格放在一起，避免標題單獨留在頁尾
        story.append(KeepTogether([cashflow_heading, make_table(rows, styles, col_widths=[200, 160, 150])]))
    else:
        story.append(cashflow_heading)

    # 四、數據品質說明
    story.append(Paragraph("四、數據品質與計算說明", styles['h1']))
    notes = [f"數據完整性：{quality.get('數據完整性')}（財報 {quality.get('財報季數')} 季）"]
    notes += quality.get('數據警告', [])
    notes += [f"缺失欄位：{item}" for item in quality.get('缺失欄位', [])]
    notes += quality.get('資料轉換說明', [])
    notes += quality.get('計算欄位說明', [])
    for note in notes:
        story.append(Paragraph(escape_text(note), styles['bullet'], bulletText='•'))

    # 五、AI 財務分析報告
    story.append(Paragraph("五、AI 財務分析報告", styles['h1']))
    ai_analysis = report.get('ai_analysis')
    if ai_analysis:
        story.append(Paragraph(escape_text(f"AI 模型：{report.get('ai_model', 'N/A')}"), styles['small']))
        story.extend(markdown_to_flowables(ai_analysis, styles))
    else:
        story.append(Paragraph("本次未執行 AI 分析（未輸入 OpenAI API 金鑰）。", styles['body']))

    # 附錄：近 8 季三大報表
    appendix_heading = Paragraph("附錄：近 8 季財務數據（單位：億元）", styles['h1'])
    for i, (subtitle, df) in enumerate([('損益表（單季）', report['income_df']), ('資產負債表（季末）', report['balance_df']),
                                        ('財務比率', report['ratio_df']), ('現金流量表（單季）', report['cash_df'])]):
        block = [Paragraph(subtitle, styles['h3']), make_table(dataframe_rows(df), styles)]
        # 附錄標題與第一張表放在一起，避免標題單獨留在頁尾
        story.append(KeepTogether(([appendix_heading] if i == 0 else []) + block))
        story.append(Spacer(1, 6))

    # 免責聲明
    story.append(Paragraph("免責聲明", styles['h2']))
    story.append(Paragraph(
        "本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，<b>不構成投資建議或財務建議</b>。"
        "請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。",
        styles['body']))

    doc.build(story, onFirstPage=draw_page_frame, onLaterPages=draw_page_frame)
    return buffer.getvalue()
