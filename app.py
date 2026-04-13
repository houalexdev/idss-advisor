"""
IDSS Insurance Decision Support System
生产版 — NiceGUI 主界面

运行：
  pip install nicegui chromadb pdfplumber python-dotenv
  MOONSHOT_API_KEY=sk-xxx python app.py

或无 API Key 演示模式：
  python app.py
"""
import os
import sys
import re
import time
import asyncio
import threading

# 禁用 ChromaDB 遥测
os.environ["CHROMA_TELEMETRY_ENABLED"]      = "false"
os.environ["CHROMADB_ANONYMIZED_TELEMETRY"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 读取环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nicegui import ui
from agents   import Orchestrator
from mock_llm import LLMConfig, LLMBackend, probe_ollama

# ── 自动检测 API Key（优先级：DeepSeek > Kimi > Mock）────────────────────────
_DS_KEY   = os.getenv("DEEPSEEK_API_KEY",  "").strip()
_KIMI_KEY = os.getenv("MOONSHOT_API_KEY",  "").strip()

if _DS_KEY:
    LLMConfig.set_backend(LLMBackend.DEEPSEEK, api_key=_DS_KEY)
    _KEY = _DS_KEY
    print(f"✅ 检测到 DEEPSEEK_API_KEY，已自动切换到 DeepSeek 模式")
elif _KIMI_KEY:
    LLMConfig.set_backend(LLMBackend.KIMI, api_key=_KIMI_KEY)
    _KEY = _KIMI_KEY
    print(f"✅ 检测到 MOONSHOT_API_KEY，已自动切换到 Kimi 模式")
else:
    _KEY = ""
    print("ℹ️  未检测到 API Key，运行 Mock 演示模式")

# ── 全局单例 ─────────────────────────────────────────────────────────────────
orch = Orchestrator()

# ── 线程安全 UI 调度 ──────────────────────────────────────────────────────────
# NiceGUI UI 操作必须在主事件循环执行，子线程通过 _ui_call() 调度
_main_loop: asyncio.AbstractEventLoop | None = None

def _ui_call(fn):
    """将 fn 安全地投递到主事件循环执行，吞掉 client 已销毁的异常"""
    global _main_loop
    def _safe():
        try:
            fn()
        except Exception:
            pass   # client 已关闭/刷新时静默忽略
    if _main_loop and _main_loop.is_running():
        _main_loop.call_soon_threadsafe(_safe)
    else:
        _safe()

# ── 常量 ─────────────────────────────────────────────────────────────────────
RED = "#C8102E"

PRODUCTS = {
    "IDSS1": ("一号产品",  "终身 · 58种 · 最高280%"),
    "IDSS2": ("二号产品",  "终身 · 115种 · 最高1100%"),
    "IDSS3": ("三号产品",  "至100岁 · 100种 · 最高100%"),
    "IDSS4": ("四号产品",  "定期 · 115种 · 最高100%"),
    "IDSS5":  ("五号产品", "终身 · 115种 · 最高900%"),
}
PROD_COLORS = {
    "IDSS1": "#b71c1c", "IDSS2": "#1565c0",
    "IDSS3": "#2e7d32", "IDSS4": "#6a1b9a", "IDSS5": "#e65100",
}

PDF_URLS = {
    "IDSS1": "/static/pdfs/01.pdf",
    "IDSS2": "/static/pdfs/02.pdf",
    "IDSS3": "/static/pdfs/03.pdf",
    "IDSS4": "/static/pdfs/04.pdf",
    "IDSS5":  "/static/pdfs/05.pdf",
}
QUICK_QS = [
    ("🎗 癌症",    "哪个计划的癌症保障最好，可以多次赔偿吗？"),
    ("❤️ 心脏",   "心脏病保障哪个最全面？"),
    ("👶 儿童",    "哪个计划的严重儿童疾病保障最好？"),
    ("💊 糖尿病",  "如果我有糖尿病风险，应该选哪个计划？"),
    ("💰 储蓄",    "哪个计划储蓄分红回报最好？"),
    ("🔁 多重",    "多重赔偿哪个计划最强？"),
    ("🏥 ICU",    "哪个计划有ICU保障？"),
    ("🧬 先天",    "先天疾病保障哪个最好？"),
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: #eef1f7;
  font-family: 'PingFang SC','Microsoft YaHei','Noto Sans SC',sans-serif;
  font-size: 13px;
  overflow-x: hidden;
}
/* 去掉 NiceGUI 默认给 body 加的 padding */
body > .q-page-container,
body > div > .q-page-container { padding: 0 !important; }
.nicegui-content { padding: 0 !important; }

/* ── Header ── */
.hdr {
  background: linear-gradient(135deg, #C8102E 0%, #7b0000 100%);
  color: #fff;
  padding: 0 20px;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.3);
  width: 100vw;
  position: fixed;
  top: 0; left: 0;
  z-index: 1000;
  margin: 0;
}
.logo        { font-size: 18px; font-weight: 900; letter-spacing: -1px; white-space: nowrap; }
.hdr-title   { display: flex; flex-direction: column; flex:1; min-width:0; overflow:hidden; }
.hdr-title .t1 { font-size: 13px; font-weight: 700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hdr-title .t2 { font-size: 10px; opacity: .7; }
.hdr-tags    { flex-shrink:0; display: flex; gap: 6px; align-items: center; }
.hdr-spacer  { height: 52px; width: 100%; flex-shrink: 0; }

/* ── Page body ── */
.page-body {
  display: grid;
  grid-template-columns: 245px 1fr 340px;
  gap: 12px;
  padding: 12px;
  box-sizing: border-box;
  width: 100%;
  min-height: calc(100vh - 52px);
  align-items: start;
}
.left-col  { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.mid-col   { display: flex; flex-direction: column; gap: 10px; min-width: 0;
             min-height: calc(100vh - 76px); }
.right-col { display: flex; flex-direction: column; gap: 10px; min-width: 0;
             min-height: calc(100vh - 76px); }
/* 中栏对比表卡片撑满 */
.mid-col > .card-table { flex: 1; display: flex; flex-direction: column; }
/* 右栏日志卡片撑满 */
.log-card { flex: 1; display: flex; flex-direction: column; min-height: 200px; }

/* ── Cards ── */
.card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
  padding: 12px;
  width: 100%;
  box-sizing: border-box;
}
.sec {
  font-size: 10.5px; font-weight: 700;
  color: #999; text-transform: uppercase;
  letter-spacing: .5px; margin-bottom: 8px;
}

/* ── Agent tags ── */
.atag { display:inline-block; padding:2px 7px; border-radius:10px; font-size:10.5px; font-weight:600; white-space:nowrap; }
.t-orch   { background:#e3f2fd; color:#1565c0 }
.t-pdf    { background:#f3e5f5; color:#6a1b9a }
.t-rag    { background:#e8f5e9; color:#2e7d32 }
.t-cmp    { background:#fff3e0; color:#b45309 }
.t-llm    { background:#fce4ec; color:#880e4f }
.t-mock     { background:#e8eaf6; color:#283593 }
.t-kimi     { background:#fff3e0; color:#b45309 }
.t-deepseek { background:#e8f5e9; color:#1b5e20 }
.t-ollama   { background:#e0f2f1; color:#004d40 }

/* ── Status dots ── */
.sdot { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }
.sg   { background:#4caf50 } .so { background:#ff9800 } .sr { background:#f44336 }

/* ── Skills chips ── */
.skchip { display:inline-block; background:#f3f4f6; border-radius:20px; padding:2px 8px; font-size:10.5px; color:#555; margin:2px; border:1px solid #e5e7eb; }

/* ── Dynamic dim badge ── */
.dyn-badge { display:inline-block; background:#006064; color:#fff; border-radius:4px; padding:2px 7px; font-size:10px; font-weight:700; margin:2px; }

/* ── Comparison table ── */
.cmp-wrap {
  overflow-x: auto;
  overflow-y: auto;
  flex: 1;
  min-height: 300px;
  max-height: calc(100vh - 180px);
  border-radius: 6px;
  width: 100%;
}
.cmp-tbl {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
  font-size: 12.5px;
}
.cmp-tbl thead th {
  padding: 10px 8px;
  text-align: center;
  position: sticky;
  top: 0;
  z-index: 3;
  font-size: 12px;
  word-break: break-word;
  line-height: 1.4;
}
.cmp-tbl td {
  padding: 8px 8px 4px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
  word-break: break-word;
}
.cmp-tbl tr:hover td { background:rgba(0,0,0,.015)!important; }
.col-label {
  position: sticky; left: 0; background: inherit;
  font-weight: 600; color: #333;
  z-index: 1; min-width: 110px; vertical-align: middle;
}
.grp-hdr td {
  background: #f1f3f9!important; color: #555;
  font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .4px;
  padding: 5px 10px!important;
}
.grp-hdr td:first-child { position:sticky; left:0; z-index:2; background:#f1f3f9!important; }
.row-diff { background:#fff8e1; }
.row-dyn  { background:#e0f7fa; border-left:3px solid #006064; }
/* 单元格内容层 */
.cell-val  { text-align: center; }
.val-text  { font-size: 12.5px; line-height: 1.4; }
.val-best  { font-weight: 700; color: #C8102E; }
.val-warn  { color: #f57c00; }
.val-no    { color: #bbb; }

/* ── 引用来源 —— 嵌入单元格底部 ── */
.cite-tag {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
  margin-top: 5px;
  padding-top: 4px;
  border-top: 1px dashed #e8e8e8;
}
.cite-sec {
  font-size: 9.5px;
  color: #999;
  font-style: italic;
  line-height: 1.2;
}
.cite-page {
  font-size: 9.5px;
  font-weight: 700;
  color: #C8102E;
  background: #fff5f5;
  border: 1px solid #fecaca;
  border-radius: 3px;
  padding: 0 4px;
  line-height: 1.5;
  white-space: nowrap;
  flex-shrink: 0;
}

/* ── Metric row ── */
.metric-row  { display:flex; gap:8px; width:100%; margin-bottom:8px; }
.metric-card { flex:1; border-radius:8px; padding:10px 8px; text-align:center; }
.metric-val  { font-size:24px; font-weight:800; }
.metric-lbl  { font-size:10px; color:#888; margin-top:2px; }

/* ── LLM panel ── */
.llm-panel { border:1px solid #e0e0e0; border-radius:8px; overflow:hidden; }
.llm-row   { display:flex; align-items:center; gap:8px; padding:8px 10px; border-bottom:1px solid #f0f0f0; cursor:pointer; transition:background .12s; }
.llm-row:last-child { border-bottom:none; }
.llm-row:hover  { background:#f9f9f9; }
.llm-row.active { background:#fff5f5; }
.llm-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }

/* ── Chat bubbles ── */
.chat-wrap  { display:flex; flex-direction:column; padding:2px 0; width:100%; }
.chat-u-row { display:flex; justify-content:flex-end; width:100%; margin:3px 0; padding-right:2px; }
.chat-u     { background:#C8102E; color:#fff; border-radius:14px 14px 3px 14px;
              padding:8px 12px; width:92%;
              font-size:12.5px; line-height:1.5; word-break:break-word;
              text-align:left; box-sizing:border-box; }
.chat-b     { background:#fff; border:1px solid #e8e8e8; border-radius:14px 14px 14px 3px;
              padding:9px 13px; max-width:96%; font-size:12.5px; line-height:1.7;
              margin:3px 0; box-shadow:0 1px 4px rgba(0,0,0,.06); word-break:break-word; }
.chat-meta  { font-size:10px; color:#bbb; margin:1px 0 5px 4px; }
.chat-ndim  { background:#e0f7fa; border:1px solid #b2ebf2; border-radius:6px;
              padding:5px 10px; font-size:11px; color:#006064; margin:4px 0; }
.chat-toast { background:#f0fdf4; border:1px solid #86efac; border-radius:6px;
              padding:4px 10px; font-size:11px; color:#15803d; margin:2px 0; }
.chat-err   { background:#fff3f3; border:1px solid #ffcdd2; border-radius:6px;
              padding:6px 10px; font-size:12px; color:#c62828; margin:2px 0; }

/* ── Log rows ── */
.log-row    { display:flex; align-items:flex-start; gap:5px; padding:3px 0; border-bottom:1px solid #f7f7f7; }
.log-detail { font-size:11px; color:#555; flex:1; line-height:1.35; word-break:break-all; }
.log-time   { font-size:10px; color:#bbb; flex-shrink:0; white-space:nowrap; }

/* ── Legend ── */
.legend { margin-top:8px; display:flex; flex-wrap:wrap; align-items:center;
          gap:8px; font-size:11px; color:#777; }
.lbox   { width:12px; height:9px; border-radius:2px; display:inline-block;
          vertical-align:middle; margin-right:2px; }
</style>
<script>
// 全局事件委托：捕获带 data-newwin 属性的元素点击，强制新标签页打开
// NiceGUI 的 shadow DOM 会拦截 <a> 标签，用 data 属性 + document 级监听绕过
document.addEventListener('click', function(e) {
    var el = e.target.closest('[data-newwin]');
    if (el) {
        e.preventDefault();
        e.stopPropagation();
        var url = el.getAttribute('data-newwin');
        if (url && url !== '#') window.open(url, '_blank');
    }
}, true);
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def md2html(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if line.startswith("## "):
            out.append(f'<p style="font-weight:700;font-size:13.5px;color:#1a1a1a;margin:6px 0 2px">{line[3:]}</p>')
        elif line.startswith("### "):
            out.append(f'<p style="font-weight:700;color:#C8102E;margin:5px 0 1px">{line[4:]}</p>')
        elif line.startswith("---"):
            out.append('<hr style="border:none;border-top:1px solid #eee;margin:6px 0">')
        elif line.startswith("- "):
            out.append(f'<li style="margin-left:14px;margin-bottom:2px">{line[2:]}</li>')
        elif line.strip() == "":
            out.append('<div style="height:4px"></div>')
        else:
            out.append(f'<p style="margin:2px 0">{line}</p>')
    html = "\n".join(out)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*',    r'<em>\1</em>',          html)
    return html


def val_class(val: str) -> str:
    if any(k in val for k in ["⭐", "✅", "市场首创", "最多", "全面", "强", "唯一"]):
        return "val-best"
    if any(k in val for k in ["⚠️", "涨价", "较长", "较严", "递增"]):
        return "val-warn"
    if any(k in val for k in ["❌", "不涵盖", "无"]):
        return "val-no"
    return ""


def build_table_html(result: dict, expanded_groups: set) -> str:
    codes = result["codes"]
    rows  = result["rows"]
    n     = len(codes)

    # 表头：维度列 + 每个产品列，无独立来源列，各列宽度均分
    dim_w = max(15, 35 - n * 3)   # 维度列占比随产品数自适应
    prod_w = (100 - dim_w) // n

    hcells = "".join(
        f'<th style="background:{PROD_COLORS.get(c, RED)};color:#fff;width:{prod_w}%">'
        f'{PRODUCTS.get(c, (c,""))[0]}<br>'
        f'<span style="font-size:10px;opacity:.8;font-weight:400">{c}</span><br>'
        f'<span data-newwin="{PDF_URLS.get(c, "#")}" '
        f'style="font-size:9.5px;color:rgba(255,255,255,.85);text-decoration:none;'
        f'background:rgba(255,255,255,.2);border-radius:3px;padding:1px 6px;'
        f'display:inline-block;margin-top:3px;cursor:pointer">📄 查看原文 PDF</span>'
        f'</th>'
        for c in codes
    )
    html = (
        '<div class="cmp-wrap"><table class="cmp-tbl"><thead><tr>'
        f'<th style="background:#2d3748;color:#fff;width:{dim_w}%;'
        'text-align:left;position:sticky;top:0;left:0;z-index:4">对比维度</th>'
        + hcells
        + '</tr></thead><tbody>'
    )

    current_group = None
    import re as _re

    for i, row in enumerate(rows):
        group   = row.get("group", "")
        primary = row.get("primary", True)
        is_diff = row["is_diff"]
        is_dyn  = row.get("dynamic", False)

        # 分组标题行
        if group != current_group:
            current_group = group
            ext_count = sum(
                1 for r in rows
                if r.get("group") == group
                and not r.get("primary")
                and not r.get("dynamic")
            )
            is_open  = group in expanded_groups
            arrow    = "▼" if is_open else "▶"
            colspan  = n + 1
            grp_icon = "★ " if "AI" in group else ""
            ext_note = (
                f" <span style='float:right;font-size:10px;color:#aaa;font-weight:400'>"
                f"{arrow} {'收起' if is_open else '展开'} {ext_count} 个扩展维度</span>"
                if ext_count else ""
            )
            html += (
                f'<tr class="grp-hdr">'
                f'<td colspan="{colspan}">{grp_icon}{group}{ext_note}</td></tr>'
            )

        show      = primary or is_dyn or (current_group in expanded_groups)
        vis_style = "" if show else 'style="display:none"'
        row_cls   = "row-dyn" if is_dyn else ("row-diff" if is_diff else "")
        bg_alt    = "" if row_cls else ("background:#fafafa;" if i % 2 == 0 else "")

        # 维度标签
        if is_dyn:
            label_html = (
                f'<span class="dyn-badge">★ AI</span>'
                f'<span style="color:#006064;font-weight:700"> {row["label"]}</span>'
            )
        elif is_diff:
            label_html = f'⚡ <span style="color:#92400e">{row["label"]}</span>'
        else:
            label_html = row["label"]

        # 每个产品单元格：值（上）+ 来源引用（下，小字斜体）
        cells = ""
        for code in codes:
            val  = str(row["values"].get(code, "N/A"))
            vcls = val_class(val)

            cite_raw = row["citation"].get(code, "")
            cite_html = ""
            if cite_raw:
                # 格式：「产品名小册子 §章节 (p.X)」→ 只取 §章节 和 (p.X) 两部分分行显示
                m_sec  = _re.search(r'§([^(]+)', cite_raw)
                m_page = _re.search(r'\(p\.([^)]+)\)', cite_raw)
                if m_sec and m_page:
                    sec  = m_sec.group(1).strip()
                    page = m_page.group(1).strip()
                    cite_html = (
                        f'<div class="cite-tag">'
                        f'<span class="cite-sec">{sec}</span>'
                        f'<span class="cite-page">p.{page}</span>'
                        f'</div>'
                    )
                else:
                    cite_html = f'<div class="cite-tag">{cite_raw}</div>'

            cells += (
                f'<td class="cell-val {vcls}">'
                f'<div class="val-text">{val}</div>'
                f'{cite_html}'
                f'</td>'
            )

        html += (
            f'<tr class="{row_cls}" style="{bg_alt}" {vis_style}>'
            f'<td class="col-label">{label_html}</td>'
            f'{cells}'
            f'</tr>'
        )

    html += (
        "</tbody></table></div>"
        '<div class="legend">'
        '<span><span class="lbox" style="background:#fff8e1;border:1px solid #fde68a"></span>'
        '⚡ 差异字段</span>'
        '<span><span class="lbox" style="background:#e0f7fa;border:1px solid #80deea"></span>'
        '★ AI动态追加</span>'
        '<span><b style="color:#C8102E">■</b> 该项突出 &nbsp;'
        '<b style="color:#bbb">■</b> 不涵盖 &nbsp;'
        '<span style="color:#aaa;font-style:italic">📎 §章节 (页码)</span></span>'
        '</div>'
    )
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 页面
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
def main_page():
    ui.add_head_html(CSS)

    state = {
        "selected":        [],
        "tbl_c":           None,
        "metrics_c":       None,
        "chat_c":          None,
        "log_c":           None,
        "log_lbl":         None,
        "dim_c":           None,
        "status_lbl":      None,
        "llm_status_lbl":  None,
        "expanded_groups": set(),
        "sel_lbl":         None,
        "go_btn":          None,
        "checks":          {},
        "chat_input":      None,
        "rebuild_btn":     None,
        "rebuild_log":     None,
        "rebuild_msgs":    [],   # 进度消息列表
    }

    # ══════════════════════════════════════════════════════════════════════
    # 辅助函数（全部在 UI 构建前定义）
    # ══════════════════════════════════════════════════════════════════════

    def on_sel():
        sel = [c for c, cb in state["checks"].items() if cb.value]
        # 最多选 5 个
        if len(sel) > 5:
            for code, cb in state["checks"].items():
                if code not in sel[:5]:
                    cb.value = False
            sel = sel[:5]
        state["selected"] = sel
        orch.set_selected_products(sel)
        n = len(sel)
        if n >= 2:
            state["sel_lbl"].set_text(f"✓ 已选 {n} 个产品")
            state["sel_lbl"].style("font-size:11px;color:#2e7d32;margin-top:4px")
        else:
            hint = f"请再选 {2-n} 个" if n == 1 else "请选择至少 2 个产品"
            state["sel_lbl"].set_text(hint)
            state["sel_lbl"].style("font-size:11px;color:#aaa;margin-top:4px")
        state["go_btn"].set_enabled(n >= 2)

    # ── 日志和维度：用 ui.timer + column.clear() 轮询刷新 ─────────────────────
    # 测试确认方式A（clear+重建）在本版本 NiceGUI 上可靠工作
    _last_log_count = [0]   # 列表以便闭包修改

    TAG = {
        "Orchestrator":  "t-orch",
        "PDF解析Agent":  "t-pdf",
        "RAG检索Agent":  "t-rag",
        "对比分析Agent": "t-cmp",
    }

    def _poll_logs():
        try:
            if state["log_lbl"] is None:
                return
            logs = orch.get_all_logs()
            if len(logs) == _last_log_count[0]:
                return
            _last_log_count[0] = len(logs)
            if not logs:
                state["log_lbl"].set_text("暂无日志")
            else:
                lines = []
                for log in reversed(logs[-30:]):
                    ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
                    lines.append(f"{log.agent}  [{log.action}] {log.detail[:55]}  {ts}")
                state["log_lbl"].set_text("\n".join(lines))
        except Exception as e:
            print(f"[LOG ERROR] {e}")

    def refresh_logs():
        _last_log_count[0] = -1
        _poll_logs()

    def refresh_dims():
        state["dim_c"].clear()
        with state["dim_c"]:
            if not orch.dynamic_dims:
                ui.label("暂无").style("font-size:11px;color:#ddd")
            else:
                for d in orch.dynamic_dims:
                    ui.html(f'<span class="dyn-badge">★ {d["name"]}</span>')

    def refresh_llm_badge():
        tag_map = {
            LLMBackend.MOCK:     "t-mock",
            LLMBackend.KIMI:     "t-kimi",
            LLMBackend.DEEPSEEK: "t-deepseek",
            LLMBackend.OLLAMA:   "t-ollama",
        }
        lbl_map = {
            LLMBackend.MOCK:     "Mock",
            LLMBackend.KIMI:     "Kimi",
            LLMBackend.DEEPSEEK: "DeepSeek",
            LLMBackend.OLLAMA:   "Ollama",
        }
        b = LLMConfig.backend
        state["llm_status_lbl"].set_content(
            f'<span class="atag {tag_map[b]}">'
            f'{lbl_map[b]}: {LLMConfig.get_effective_model()}</span>'
        )

    def render_metrics(result: dict):
        state["metrics_c"].clear()
        stats = result.get("stats", {})
        with state["metrics_c"]:
            ui.html(
                '<div class="metric-row">'
                + "".join(
                    f'<div class="metric-card" style="background:{bg}">'
                    f'<div class="metric-val" style="color:{fg}">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div></div>'
                    for val, lbl, bg, fg in [
                        (str(len(result["codes"])),       "已选产品",   "#fff5f5", RED),
                        (str(stats.get("total_dims", 0)), "对比维度",   "#f0f4ff", "#1565c0"),
                        (str(stats.get("diff_count", 0)), "差异字段",   "#fffbe6", "#92400e"),
                        (str(stats.get("dyn_count",  0)), "AI动态维度", "#e0f7fa", "#006064"),
                    ]
                )
                + '</div>'
            )

    def render_table(result: dict):
        state["tbl_c"].clear()
        with state["tbl_c"]:
            ui.html(build_table_html(result, state["expanded_groups"]))

    def append_chat(role: str, text: str = "", sources=None,
                    chunks: int = 0, new_dim=None, error: str = ""):
        with state["chat_c"]:
            if role == "user":
                ui.html(
                    f'<div class="chat-u-row">'
                    f'<div class="chat-u">{text}</div></div>'
                )
            else:
                if error:
                    ui.html(f'<div class="chat-err">⚠️ {error}</div>')
                else:
                    body = md2html(text)
                    ui.html(f'<div class="chat-b">{body}</div>')
                    meta_parts = []
                    if sources:
                        valid_src = [s for s in sources[:2] if s]
                        if valid_src:
                            meta_parts.append("📎 " + " · ".join(valid_src))
                    if chunks:
                        meta_parts.append(f"检索 {chunks} 段")
                    if meta_parts:
                        ui.html(f'<div class="chat-meta">{" · ".join(meta_parts)}</div>')

    def do_rebuild_pdf():
        """从 PDF 用 LLM 提取结构化数据，重建 products_index.json"""
        from mock_llm import LLMConfig, LLMBackend

        if LLMConfig.backend == LLMBackend.MOCK:
            ui.notify("⚠️ 请先切换到 Kimi 或 Ollama 模式", type="warning")
            return

        # 显示进度区，禁用按钮
        state["rebuild_msgs"] = []
        state["rebuild_btn"].set_enabled(False)
        state["rebuild_btn"].set_text("⏳ 提取中…")

        def show_log():
            """在进度区显示所有消息"""
            state["rebuild_log"].clear()
            with state["rebuild_log"]:
                for msg in state["rebuild_msgs"]:
                    color = "#c62828" if msg.startswith("❌") else (
                            "#2e7d32" if msg.startswith("✅") else (
                            "#b45309" if msg.startswith("⚠️") else "#1565c0"))
                    ui.html(
                        f'<div style="font-size:10.5px;color:{color};'
                        f'padding:1.5px 6px;border-bottom:1px solid #eaf4ff">'
                        f'{msg}</div>'
                    )
            # 使进度框可见
            state["rebuild_log"].style(
                "display:block;height:120px;width:100%;margin-top:6px;"
                "border:1px solid #e3f2fd;border-radius:6px;background:#f8fbff"
            )

        def on_progress(msg: str):
            state["rebuild_msgs"].append(msg)
            _ui_call(show_log)

        def run():
            try:
                report = orch.rebuild_index_from_pdfs(on_progress=on_progress)

                def done():
                    state["rebuild_btn"].set_enabled(True)
                    state["rebuild_btn"].set_text("📄 从 PDF 重建数据")

                    ok  = sum(1 for v in report.values() if v.get("status") == "ok")
                    err = sum(1 for v in report.values() if v.get("status") == "failed")
                    skp = sum(1 for v in report.values() if v.get("status") == "skipped")

                    if "error" in report:
                        ui.notify(report["error"], type="warning")
                    else:
                        msg = f"✅ 重建完成：{ok} 成功"
                        if skp: msg += f"，{skp} 跳过（无PDF）"
                        if err: msg += f"，{err} 失败"
                        ui.notify(msg, type="positive" if not err else "warning")
                        state["status_lbl"].set_text(
                            f"✅ PDF重建完成：{ok}/{len(report)}个产品"
                        )
                        # 对比表已有选择时自动刷新
                        if len(state["selected"]) >= 2:
                            result = orch.get_comparison()
                            if result:
                                render_table(result)
                                render_metrics(result)
                    refresh_logs()

                _ui_call(done)

            except Exception as e:
                def err():
                    state["rebuild_btn"].set_enabled(True)
                    state["rebuild_btn"].set_text("📄 从 PDF 重建数据")
                    ui.notify(f"重建失败: {str(e)[:60]}", type="negative")
                _ui_call(err)

        threading.Thread(target=run, daemon=True).start()

    def do_compare():
        if len(state["selected"]) < 2:
            ui.notify("请至少选择 2 个产品", type="warning")
            return
        state["status_lbl"].set_text("🔄 正在生成对比分析…")
        state["go_btn"].set_enabled(False)

        def run():
            try:
                for code in state["selected"]:
                    orch.ensure_indexed(
                        code,
                        on_progress=lambda m: _ui_call(
                            lambda m=m: state["status_lbl"].set_text(m)
                        )
                    )
                result = orch.get_comparison()
                if not result:
                    def upd_err():
                        state["status_lbl"].set_text("❌ 对比生成失败")
                        state["go_btn"].set_enabled(True)
                    _ui_call(upd_err)
                    return

                def upd():
                    state["status_lbl"].set_text("✅ 对比表已生成")
                    state["go_btn"].set_enabled(True)
                    render_metrics(result)
                    render_table(result)
                    refresh_logs()

                _ui_call(upd)
            except Exception as e:
                err = str(e)
                def upd_err():
                    state["status_lbl"].set_text(f"❌ {err[:60]}")
                    state["go_btn"].set_enabled(True)
                    refresh_logs()
                _ui_call(upd_err)

        threading.Thread(target=run, daemon=True).start()

    def do_chat(msg: str):
        if not msg or not msg.strip():
            return
        if len(state["selected"]) < 2:
            ui.notify("请先选择至少 2 个产品并生成对比表", type="warning")
            return
        msg = msg.strip()
        # 清空输入框
        if state["chat_input"]:
            state["chat_input"].value = ""
        append_chat("user", msg)

        def run():
            try:
                result = orch.chat(msg)

                def upd():
                    append_chat(
                        "bot",
                        result["answer"],
                        sources=result.get("sources", []),
                        chunks=result.get("chunks_used", 0),
                        new_dim=result.get("new_dimension"),
                    )
                    if result.get("new_dimension"):
                        # 用聊天区内的 toast 替代 ui.notify（后者在回调中会触发 slot 错误）
                        with state["chat_c"]:
                            ui.html(
                                f'<div class="chat-toast">'
                                f'✨ 已将「{result["new_dimension"]["name"]}」新增至对比表</div>'
                            )
                        refresh_dims()
                        r2 = orch.get_comparison()
                        if r2:
                            render_table(r2)
                            render_metrics(r2)
                    refresh_logs()

                _ui_call(upd)
            except Exception as e:
                err_msg = str(e)
                def upd_err():
                    append_chat("bot", error=err_msg)
                    refresh_logs()
                _ui_call(upd_err)

        threading.Thread(target=run, daemon=True).start()

    def do_clear_logs():
        orch.own_logs.clear()
        orch.pdf_agent.logs.clear()
        orch.rag_agent.logs.clear()
        orch.compare_agent.logs.clear()
        _last_log_count[0] = -1
        if state["log_lbl"]:
            state["log_lbl"].set_text("已清空")

    def do_reset():
        orch.clear_session()
        for cb in state["checks"].values():
            cb.value = False
        state["selected"]        = []
        state["expanded_groups"] = set()
        state["sel_lbl"].set_text("请选择至少 2 个产品")
        state["sel_lbl"].style("font-size:11px;color:#aaa;margin-top:4px")
        state["go_btn"].set_enabled(False)
        state["status_lbl"].set_text("")
        state["tbl_c"].clear()
        with state["tbl_c"]:
            ui.label("← 请在左侧选择产品，然后点击「生成对比分析」").style(
                "color:#bbb;font-size:13px;padding:48px 0;text-align:center;width:100%")
        state["metrics_c"].clear()
        state["chat_c"].clear()
        with state["chat_c"]:
            ui.html('<div style="color:#bbb;font-size:12px;padding:8px">会话已重置。</div>')
        refresh_logs()
        refresh_dims()
        ui.notify("会话已重置", type="info")

    def build_llm_panel(container):
        container.clear()
        cur = LLMConfig.backend

        with container:
            with ui.element("div").classes("llm-panel"):
                for backend, color, title, sub in [
                    (LLMBackend.MOCK,     "#283593",
                     "Mock（本地知识库，无需 API）",  "适合演示，零延迟"),
                    (LLMBackend.DEEPSEEK, "#1b5e20",
                     "DeepSeek（推荐）",              "deepseek-chat / deepseek-reasoner"),
                    (LLMBackend.KIMI,     "#b45309",
                     "Kimi（Moonshot AI）",           "moonshot-v1-8k / 32k / 128k"),
                    (LLMBackend.OLLAMA,   "#004d40",
                     "Ollama（本地大模型）",           LLMConfig.ollama_url),
                ]:
                    active_cls = "active" if cur == backend else ""

                    def on_click(_, b=backend, c=container):
                        if b == LLMBackend.MOCK:
                            LLMConfig.set_backend(LLMBackend.MOCK)
                            refresh_llm_badge()
                            ui.notify("✓ 已切换到 Mock 模式", type="positive")
                        elif b == LLMBackend.DEEPSEEK:
                            LLMConfig.backend = LLMBackend.DEEPSEEK
                        elif b == LLMBackend.KIMI:
                            LLMConfig.backend = LLMBackend.KIMI
                        elif b == LLMBackend.OLLAMA:
                            LLMConfig.backend = LLMBackend.OLLAMA
                        build_llm_panel(c)

                    with ui.row().classes(f"llm-row {active_cls}").on("click", on_click):
                        ui.html(f'<span class="llm-dot" style="background:{color}"></span>')
                        with ui.column().style("gap:0;flex:1"):
                            ui.label(title).style("font-size:12px;font-weight:600")
                            ui.label(sub).style("font-size:10px;color:#888")
                        if cur == backend:
                            ui.html('<span style="color:#C8102E;font-size:14px;font-weight:700">✓</span>')

            # ── DeepSeek 配置 ──
            if cur == LLMBackend.DEEPSEEK:
                with ui.element("div").style(
                        "border:1px solid #c8e6c9;border-radius:8px;"
                        "padding:10px;background:#f1f8f1;margin-top:6px"):
                    ui.label("DeepSeek 配置").style(
                        "font-size:11px;font-weight:700;color:#1b5e20;margin-bottom:6px")
                    ds_key_inp = ui.input(
                        "DEEPSEEK_API_KEY",
                        value=LLMConfig.api_key or "",
                        password=True, placeholder="sk-…"
                    ).props("dense outlined").classes("w-full")
                    ds_model_inp = ui.input(
                        "模型", value=LLMConfig.get_effective_model(),
                        placeholder="deepseek-chat"
                    ).props("dense outlined").classes("w-full").style("margin-top:4px")
                    ui.html(
                        '<div style="font-size:10px;color:#888;margin-top:4px">'
                        '可选模型：deepseek-chat（V3，推荐）/ deepseek-reasoner（R1）</div>'
                    )

                    def do_apply_deepseek(_=None, c=container):
                        k = ds_key_inp.value.strip()
                        if not k:
                            ui.notify("请输入 API Key", type="warning")
                            return
                        LLMConfig.set_backend(
                            LLMBackend.DEEPSEEK,
                            api_key=k,
                            model=ds_model_inp.value or "deepseek-chat"
                        )
                        ui.notify(f"✓ 已切换到 DeepSeek ({LLMConfig.get_effective_model()})",
                                  type="positive")
                        refresh_llm_badge()
                        build_llm_panel(c)

                    ui.button("应用", icon="check", on_click=do_apply_deepseek).classes("w-full").style(
                        "background:#1b5e20;color:#fff;font-size:12px;border-radius:6px;margin-top:6px")
                    ui.html(
                        '<a href="https://platform.deepseek.com" target="_blank" '
                        'style="font-size:10px;color:#888;display:block;margin-top:4px">'
                        '获取 API Key → platform.deepseek.com</a>'
                    )

            # ── Kimi 配置 ──
            if cur == LLMBackend.KIMI:
                with ui.element("div").style(
                        "border:1px solid #ffe0b2;border-radius:8px;"
                        "padding:10px;background:#fffbf5;margin-top:6px"):
                    ui.label("Kimi 配置").style(
                        "font-size:11px;font-weight:700;color:#b45309;margin-bottom:6px")
                    key_inp = ui.input(
                        "MOONSHOT_API_KEY",
                        value=LLMConfig.api_key or "",
                        password=True, placeholder="sk-…"
                    ).props("dense outlined").classes("w-full")
                    model_inp = ui.input(
                        "模型", value=LLMConfig.get_effective_model(),
                        placeholder="moonshot-v1-8k"
                    ).props("dense outlined").classes("w-full").style("margin-top:4px")

                    def do_apply_kimi(_=None, c=container):
                        k = key_inp.value.strip()
                        if not k:
                            ui.notify("请输入 API Key", type="warning")
                            return
                        LLMConfig.set_backend(
                            LLMBackend.KIMI,
                            api_key=k,
                            model=model_inp.value or "moonshot-v1-8k"
                        )
                        ui.notify(f"✓ 已切换到 Kimi ({LLMConfig.get_effective_model()})",
                                  type="positive")
                        refresh_llm_badge()
                        build_llm_panel(c)

                    ui.button("应用", icon="check", on_click=do_apply_kimi).classes("w-full").style(
                        "background:#b45309;color:#fff;font-size:12px;border-radius:6px;margin-top:6px")
                    ui.html(
                        '<a href="https://platform.moonshot.cn" target="_blank" '
                        'style="font-size:10px;color:#888;display:block;margin-top:4px">'
                        '获取 API Key → platform.moonshot.cn</a>'
                    )

            # ── Ollama 配置 ──
            if cur == LLMBackend.OLLAMA:
                with ui.element("div").style(
                        "border:1px solid #e0f2f1;border-radius:8px;"
                        "padding:10px;background:#f5fffe;margin-top:6px"):
                    ui.label("Ollama 配置").style(
                        "font-size:11px;font-weight:700;color:#004d40;margin-bottom:6px")
                    url_inp = ui.input(
                        "API 地址", value=LLMConfig.ollama_url
                    ).props("dense outlined").classes("w-full")
                    # 默认填入推荐模型
                    _cur_model = LLMConfig.get_effective_model()
                    _default_model = _cur_model if _cur_model != "mock-local" else "qwen3:30b"
                    model_inp2 = ui.input(
                        "模型名称", value=_default_model,
                        placeholder="qwen3:30b"
                    ).props("dense outlined").classes("w-full").style("margin-top:4px")
                    probe_lbl = ui.label("").style("font-size:11px;margin-top:2px")

                    def do_probe(_=None):
                        probe_lbl.set_text("🔍 探测中…")
                        probe_lbl.style("color:#555")
                        def run():
                            r = probe_ollama(url_inp.value)
                            def upd():
                                if r["ok"]:
                                    probe_lbl.set_text(
                                        f"✓ 发现 {len(r['models'])} 个模型："
                                        + ", ".join(r["models"][:4]))
                                    probe_lbl.style("color:#004d40")
                                else:
                                    probe_lbl.set_text(f"✗ 连接失败：{r.get('error','')[:50]}")
                                    probe_lbl.style("color:#c62828")
                            _ui_call(upd)
                        threading.Thread(target=run, daemon=True).start()

                    def do_apply_ollama(_=None, c=container):
                        LLMConfig.set_backend(
                            LLMBackend.OLLAMA,
                            model=model_inp2.value,
                            ollama_url=url_inp.value
                        )
                        ui.notify(f"✓ 已切换到 Ollama ({model_inp2.value})", type="positive")
                        refresh_llm_badge()
                        build_llm_panel(c)

                    with ui.row().style("gap:6px;margin-top:6px"):
                        ui.button("探测模型", icon="search", on_click=do_probe).style(
                            "background:#004d40;color:#fff;border-radius:6px;font-size:12px;flex:1"
                        ).props("dense")
                        ui.button("应用", icon="check", on_click=do_apply_ollama).style(
                            "background:#004d40;color:#fff;border-radius:6px;font-size:12px"
                        ).props("dense")

    # ══════════════════════════════════════════════════════════════════════
    # UI 构建
    # ══════════════════════════════════════════════════════════════════════

    # ── HEADER ────────────────────────────────────────────────────────────
    ui.html(
        '<div class="hdr">'
        '<span class="logo">IDSS Advisor</span>'
        '<div class="hdr-title">'
        '<span class="t1">保险产品智能决策支持系统</span>'
        '<span class="t2">AI-Powered Insurance Decision Support System</span>'
        '</div>'
        '<div class="hdr-tags">'
        '<span class="atag t-orch">Orchestrator</span>'
        '<span class="atag t-pdf">PDF解析Agent</span>'
        '<span class="atag t-rag">RAG检索Agent</span>'
        '<span class="atag t-cmp">对比分析Agent</span>'
        '<span class="atag t-llm">LLM</span>'
        '</div>'
        '</div>'
        '<div class="hdr-spacer"></div>'
    )

    # ── BODY ──────────────────────────────────────────────────────────────
    with ui.element("div").classes("page-body"):

        # ══ 左栏 ══════════════════════════════════════════════════════════
        with ui.element("div").classes("left-col"):

            # 产品选择
            with ui.element("div").classes("card"):
                ui.html('<div class="sec">① 选择产品（2–5个）</div>')
                for code, (name, desc) in PRODUCTS.items():
                    with ui.row().style("align-items:flex-start;gap:6px;padding:3px 0"):
                        cb = ui.checkbox(value=False).props("dense")
                        cb.on("update:model-value", lambda _: on_sel())
                        state["checks"][code] = cb
                        with ui.column().style("gap:0;flex:1;padding-top:1px"):
                            ui.label(name).style("font-weight:600;font-size:13px;line-height:1.3")
                            ui.label(f"{code} · {desc}").style("font-size:10px;color:#888")
                state["sel_lbl"] = ui.label("请选择至少 2 个产品").style(
                    "font-size:11px;color:#aaa;margin-top:4px")

            # 生成按钮
            state["go_btn"] = ui.button(
                "② 生成对比分析 →",
                icon="compare_arrows",
                on_click=do_compare,
            ).style(
                f"background:{RED};color:#fff;font-weight:700;"
                "border-radius:8px;font-size:13px;width:100%;padding:10px 0"
            )
            state["go_btn"].set_enabled(False)

            # 系统状态
            with ui.element("div").classes("card"):
                ui.html('<div class="sec">系统状态</div>')
                _key_status = ("sg", f"API Key: {_KEY[:8]}…" if _KEY else "未设置（Mock 模式）")
                for dot, txt in [
                    ("sg", "ChromaDB 向量库已就绪"),
                    ("sg", "5个产品结构化数据已加载"),
                    _key_status,
                ]:
                    ui.html(
                        f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px">'
                        f'<span class="sdot {dot}"></span>'
                        f'<span style="font-size:11px;color:#555">{txt}</span>'
                        f'</div>'
                    )
                state["status_lbl"] = ui.label("").style(
                    "font-size:11px;color:#1565c0;margin-top:2px")

                # ── PDF 重建按钮 ────────────────────────────────────────────
                ui.html(
                    '<div style="border-top:1px solid #f0f0f0;margin:8px 0 6px"></div>'
                    '<div style="font-size:10.5px;color:#888;margin-bottom:6px">'
                    '从 PDF 提取真实数据</div>'
                )
                state["rebuild_btn"] = ui.button(
                    "📄 从 PDF 重建数据", on_click=do_rebuild_pdf
                ).style(
                    "background:#1565c0;color:#fff;border-radius:6px;"
                    "font-size:11.5px;width:100%;padding:6px 0"
                ).props("dense")

                # 重建进度日志
                state["rebuild_log"] = ui.scroll_area().style(
                    "display:none;height:120px;width:100%;margin-top:6px;"
                    "border:1px solid #e3f2fd;border-radius:6px;background:#f8fbff"
                )
                with state["rebuild_log"]:
                    ui.label("").style("font-size:10.5px;color:#555;padding:4px")

            # LLM 设置
            with ui.element("div").classes("card"):
                ui.html('<div class="sec">③ LLM 设置</div>')
                llm_container = ui.column().classes("w-full").style("gap:0")
                build_llm_panel(llm_container)

            ui.button("🔄 重置会话", on_click=do_reset).props("flat dense").style(
                "width:100%;font-size:11px;color:#aaa;margin-top:2px")

        # ══ 中栏 ══════════════════════════════════════════════════════════
        with ui.element("div").classes("mid-col"):

            state["metrics_c"] = ui.element("div").style("width:100%")

            with ui.element("div").classes("card card-table").style("width:100%;min-width:0;overflow:hidden"):
                ui.html(
                    '<div style="display:flex;align-items:center;'
                    'justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:6px">'
                    '<span style="font-size:15px;font-weight:700;white-space:nowrap">'
                    '保险产品全面对比表</span>'
                    '<span style="font-size:10.5px;color:#aaa">'
                    '⚡ 差异高亮 &nbsp;·&nbsp; ▶ 展开分组 &nbsp;·&nbsp; '
                    '★ AI动态追加 &nbsp;·&nbsp; 📎 引用嵌入格内</span>'
                    '</div>'
                )
                state["tbl_c"] = ui.column().classes("w-full").style("width:100%")
                with state["tbl_c"]:
                    ui.html(
                        '<div style="color:#bbb;font-size:13px;padding:60px 0;text-align:center">'
                        '← 请在左侧选择 2–5 个产品，然后点击「② 生成对比分析」</div>'
                    )

        # ══ 右栏 ══════════════════════════════════════════════════════════
        with ui.element("div").classes("right-col"):

            # 聊天面板
            with ui.element("div").classes("card"):
                # 标题行
                ui.html(
                    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                    '<span style="font-size:16px">🤖</span>'
                    '<span style="font-size:13px;font-weight:700;flex:1">智能问答助手</span>'
                    '</div>'
                )
                # 初始 badge 根据当前后端显示
                _tag_init = {
                    LLMBackend.MOCK:     "t-mock",
                    LLMBackend.KIMI:     "t-kimi",
                    LLMBackend.DEEPSEEK: "t-deepseek",
                    LLMBackend.OLLAMA:   "t-ollama",
                }.get(LLMConfig.backend, "t-mock")
                _lbl_init = {
                    LLMBackend.MOCK:     "Mock: mock-local",
                    LLMBackend.KIMI:     f"Kimi: {LLMConfig.get_effective_model()}",
                    LLMBackend.DEEPSEEK: f"DeepSeek: {LLMConfig.get_effective_model()}",
                    LLMBackend.OLLAMA:   f"Ollama: {LLMConfig.get_effective_model()}",
                }.get(LLMConfig.backend, "Mock: mock-local")
                state["llm_status_lbl"] = ui.html(
                    f'<span class="atag {_tag_init}">{_lbl_init}</span>'
                )

                # 聊天滚动区域
                chat_scroll = ui.scroll_area().style(
                    "height:310px;width:100%;box-sizing:border-box;"
                    "border:1px solid #eee;border-radius:8px;"
                    "padding:0;margin-top:8px;background:#fafafa"
                )
                with chat_scroll:
                    state["chat_c"] = ui.element("div").style(
                        "padding:8px 6px;min-height:100%"
                    )
                    with state["chat_c"]:
                        ui.html(
                            '<div style="color:#bbb;font-size:12px;padding:8px;line-height:1.7">'
                            '👋 选好产品并生成对比表后，向我提问！<br>'
                            '<span style="font-size:10.5px;color:#ccc">'
                            '例如：哪个计划癌症赔偿次数最多？</span>'
                            '</div>'
                    )

                # 输入行
                with ui.row().style(
                        "width:100%;gap:6px;margin-top:8px;align-items:center"):
                    chat_input = ui.input(placeholder="请输入问题…").classes("flex-1").props(
                        "dense outlined clearable")
                    state["chat_input"] = chat_input
                    ui.button("发送", icon="send",
                              on_click=lambda: do_chat(chat_input.value)).style(
                        f"background:{RED};color:#fff;border-radius:8px;min-width:60px"
                    ).props("dense")
                chat_input.on("keydown.enter", lambda _=None: do_chat(chat_input.value))

                # 快速提问按钮
                ui.label("快速提问：").style(
                    "font-size:10.5px;color:#bbb;margin-top:8px;margin-bottom:4px;display:block")
                with ui.row().style("flex-wrap:wrap;gap:4px"):
                    for lbl, q in QUICK_QS:
                        ui.button(lbl, on_click=lambda q=q: do_chat(q)).props(
                            "flat dense").style(
                            "background:#f0f0f0;border-radius:12px;"
                            "padding:2px 8px;font-size:11px;color:#555")

            # Agent 日志
            with ui.element("div").classes("card log-card"):
                with ui.row().style(
                        "display:flex;align-items:center;"
                        "justify-content:space-between;margin-bottom:6px"):
                    ui.label("Agent 运行日志").style("font-size:13px;font-weight:700")
                    ui.button("清空", on_click=do_clear_logs).props("flat dense").style(
                        "font-size:11px;color:#aaa")
                with ui.element("div").style(
                    "height:160px;width:100%;overflow-y:auto;"
                    "border:1px solid #f0f0f0;border-radius:8px;padding:4px"
                ):
                    state["log_c"]   = None
                    state["log_lbl"] = ui.label("等待操作…").style(
                        "font-size:11px;color:#aaa;padding:8px;"
                        "white-space:pre-wrap;word-break:break-all;width:100%;display:block"
                    )

            # Skills 注册状态
            with ui.element("div").classes("card"):
                ui.html('<div class="sec">已注册 Skills</div>')
                for s in ["pdf_parse", "semantic_chunk", "vector_index",
                          "retrieval", "table_build", "diff_detect"]:
                    ui.html(f'<span class="skchip">⚡ {s}</span>')

            # AI 动态对比维度（右栏）
            with ui.element("div").classes("card"):
                ui.html('<div class="sec">AI 动态对比维度</div>')
                ui.html('<div style="font-size:10.5px;color:#aaa;margin-bottom:6px">'
                        '问答后自动扩展对比表</div>')
                state["dim_c"] = ui.row().style("flex-wrap:wrap;gap:3px")
                with state["dim_c"]:
                    ui.label("暂无").style("font-size:11px;color:#ddd")

    # 日志每秒轮询
    ui.timer(1.0, _poll_logs)


# ─────────────────────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    import asyncio
    from nicegui import app as _app
    from pathlib import Path as _Path

    # 挂载静态 PDF 目录（/static/pdfs/ → data/ 目录）
    _data_dir = _Path(__file__).resolve().parent / "data"
    _data_dir.mkdir(exist_ok=True)
    _app.add_static_files("/static/pdfs", str(_data_dir))

    # 捕获主事件循环，供子线程的 _ui_call 使用
    try:
        _main_loop = asyncio.get_event_loop()
    except RuntimeError:
        _main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_main_loop)

    print("=" * 55)
    print("IDSS Insurance Decision Support System")
    print(f"LLM Backend : {LLMConfig.backend.value} ({LLMConfig.get_effective_model()})")
    print("访问地址    : http://localhost:8080")
    print("PDF 链接    : http://localhost:8080/static/pdfs/<filename>")
    print("=" * 55)
    ui.run(
        host="0.0.0.0",
        port=8080,
        title="保险产品智能决策系统",
        favicon="🏥",
        reload=False,
        show=False,
    )
