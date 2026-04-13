"""
Insurance Decision Support System — Multi-Agent System
架构：
  Orchestrator
    ├── PDFAgent      (Skills: pdf_parse, semantic_chunk, vector_index)
    ├── RAGAgent      (Skill: retrieval)
    └── CompareAgent  (Skills: table_build, diff_detect)

对比表：40 维度，分 5 大组，每组区分"主要"与"扩展"：
  主要维度 — 默认显示（点击可展开）
  扩展维度 — 折叠，用户点击展开
"""
import json
import re
import os
import time
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import pdfplumber

# ─────────────────────────────────────────────────────────────────────────────
# 所有路径相对于 agents.py 文件所在目录，不受工作目录影响
_BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR    = _BASE_DIR / "data"
CHROMA_DIR  = _BASE_DIR / "chroma_db"

PRODUCT_PATHS = {
    "IDSS1": DATA_DIR / "01.pdf",
    "IDSS2": DATA_DIR / "02.pdf",
    "IDSS3": DATA_DIR / "03.pdf",
    "IDSS4": DATA_DIR / "04.pdf",
    "IDSS5":  DATA_DIR / "05.pdf",
}
PRODUCT_NAMES_ZH = {
    "IDSS1": "一号产品",
    "IDSS2": "二号产品",
    "IDSS3": "三号产品",
    "IDSS4": "四号产品",
    "IDSS5": "五号产品",
}

# =============================================================================
# SKILLS
# =============================================================================

class PDFParseSkill:
    name = "pdf_parse"
    def run(self, pdf_path: str) -> dict:
        pages = {}
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                # Extract tables and append as text
                try:
                    tables = page.extract_tables() or []
                    for tbl in tables:
                        for row in tbl:
                            if row:
                                text += "\n" + " | ".join(str(c).strip() if c else "" for c in row)
                except Exception:
                    pass
                pages[i] = text
        return {"pages": pages, "total_pages": len(pages)}


class SemanticChunkSkill:
    name = "semantic_chunk"
    def run(self, pages: dict, product_code: str, product_name: str) -> dict:
        chunks = []
        for page_num, text in pages.items():
            if len(text) < 30:
                continue
            for seg in re.split(r'\n{2,}', text):
                seg = seg.strip()
                if len(seg) < 40:
                    continue
                if len(seg) > 600:
                    buf = ""
                    for part in re.split(r'[。；\n]', seg):
                        part = part.strip()
                        if not part:
                            continue
                        if len(buf) + len(part) < 500:
                            buf += part + "。"
                        else:
                            if buf:
                                chunks.append(self._c(buf, page_num, product_code, product_name))
                            buf = part + "。"
                    if buf:
                        chunks.append(self._c(buf, page_num, product_code, product_name))
                else:
                    chunks.append(self._c(seg, page_num, product_code, product_name))
        return {"chunks": chunks, "count": len(chunks)}

    def _c(self, text, page, code, name):
        return {"text": text, "page": page, "product_code": code,
                "product_name": name, "source": f"{name}（{code}），第{page}页"}


class VectorIndexSkill:
    name = "vector_index"
    def __init__(self, col): self.col = col
    def run(self, chunks: list, product_code: str) -> dict:
        try:
            ex = self.col.get(where={"product_code": product_code})
            if ex["ids"]: self.col.delete(ids=ex["ids"])
        except Exception: pass
        if not chunks: return {"indexed": 0}
        ids, docs, metas = [], [], []
        for i, c in enumerate(chunks):
            ids.append(hashlib.md5(f"{product_code}_{i}_{c['text'][:50]}".encode()).hexdigest())
            docs.append(c["text"])
            metas.append({"product_code": c["product_code"], "product_name": c["product_name"],
                          "page": c["page"], "source": c["source"]})
        for s in range(0, len(ids), 100):
            self.col.add(ids=ids[s:s+100], documents=docs[s:s+100], metadatas=metas[s:s+100])
        return {"indexed": len(ids)}


class RetrievalSkill:
    name = "retrieval"
    def __init__(self, col): self.col = col
    def run(self, query: str, product_codes: list, n_results: int = 5) -> dict:
        try:
            where = {"product_code": {"$in": product_codes}} if product_codes else None
            r = self.col.query(query_texts=[query], n_results=min(n_results, 10), where=where)
            chunks = []
            if r["documents"] and r["documents"][0]:
                for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
                    chunks.append({"text": doc, "source": meta.get("source",""),
                                   "page": meta.get("page",0),
                                   "product_code": meta.get("product_code","")})
            return {"chunks": chunks}
        except Exception as e:
            return {"chunks": [], "error": str(e)}


# =============================================================================
# TABLE BUILD SKILL — 40维度，分组，含 primary/extended 标记
# =============================================================================
#
# 维度定义格式：
#   (field_key, label, group, primary, citation_hint)
#
# primary=True  → 默认展示
# primary=False → 折叠在"展开更多"里
#
# 共 5 大组：
#   A. 产品定位    (4 维, primary 2)
#   B. 疾病覆盖    (6 维, primary 4)
#   C. 赔偿机制    (14维, primary 6)
#   D. 特色保障    (8 维, primary 4)
#   E. 财务/合同   (8 维, primary 4)
# =============================================================================

DIMENSIONS = [
    # ── A. 产品定位 ────────────────────────────────────────────────────────
    ("product_position",   "产品定位",       "A. 产品定位", True,  "产品资料"),
    ("plan_type",          "计划类型",       "A. 产品定位", True,  "产品资料"),
    ("coverage_period",    "保障年期",       "A. 产品定位", False, "产品资料"),
    ("premium_terms",      "缴费期",         "A. 产品定位", False, "产品资料"),

    # ── B. 疾病覆盖 ────────────────────────────────────────────────────────
    ("disease_total",      "受保疾病总数",   "B. 疾病覆盖", True,  "保障疾病一览表"),
    ("disease_severe",     "严重危疾种数",   "B. 疾病覆盖", True,  "保障疾病一览表"),
    ("disease_early",      "早期危疾种数",   "B. 疾病覆盖", True,  "保障疾病一览表"),
    ("disease_children",   "严重儿童疾病",   "B. 疾病覆盖", True,  "保障疾病一览表"),
    ("early_not_erode",    "早期赔付不侵蚀保额", "B. 疾病覆盖", False, "产品特点"),
    ("issue_age",          "投保年龄",       "B. 疾病覆盖", False, "产品资料"),

    # ── C. 赔偿机制 ────────────────────────────────────────────────────────
    ("payout_structure",   "赔付结构类型",   "C. 赔偿机制", True,  "保障疾病赔偿一览表"),
    ("max_payout_pct",     "最高总赔偿额",   "C. 赔偿机制", True,  "保障疾病赔偿一览表"),
    ("cancer_claims",      "癌症最多赔偿次数","C. 赔偿机制", True, "保障疾病赔偿一览表"),
    ("heart_claims",       "心脏病最多赔偿", "C. 赔偿机制", True,  "保障疾病赔偿一览表"),
    ("stroke_claims",      "中风最多赔偿",   "C. 赔偿机制", True,  "保障疾病赔偿一览表"),
    ("other_multiple",     "其他疾病多重赔", "C. 赔偿机制", True,  "保障疾病赔偿一览表"),
    ("cancer_trigger",     "癌症触发条件",   "C. 赔偿机制", False, "保障疾病赔偿一览表"),
    ("cancer_interval",    "癌症间隔期",     "C. 赔偿机制", False, "保障疾病赔偿一览表"),
    ("cancer_cashflow",    "癌症持续现金流", "C. 赔偿机制", False, "保障疾病赔偿一览表"),
    ("first_ci_payout",    "首次重疾赔付",   "C. 赔偿机制", False, "保障疾病赔偿一览表"),
    ("early_payout_pct",   "早期疾病预支比例","C. 赔偿机制", False, "保障疾病赔偿一览表"),
    ("waiting_period",     "等候期",         "C. 赔偿机制", False, "保障条款"),
    ("survival_period",    "生存期要求",     "C. 赔偿机制", False, "保障条款"),

    # ── D. 特色保障 ────────────────────────────────────────────────────────
    ("icu_coverage",       "ICU 保障",       "D. 特色保障", True,  "保障特点"),
    ("congenital",         "先天疾病保障",   "D. 特色保障", True,  "保障特点"),
    ("benign_tumor",       "良性病变保障",   "D. 特色保障", True,  "保障特点"),
    ("dementia_annuity",   "脑退化年金",     "D. 特色保障", True,  "保障特点"),
    ("diabetes_coverage",  "糖尿病/慢性病",  "D. 特色保障", False, "保障疾病一览表"),
    ("pregnancy_cover",    "孕期/儿童保障",  "D. 特色保障", False, "保障特点"),
    ("premium_waiver",     "保费豁免",       "D. 特色保障", False, "产品资料"),
    ("sa_increase",        "保额提升机制",   "D. 特色保障", False, "产品特点"),

    # ── E. 财务/合同 ────────────────────────────────────────────────────────
    ("savings_attr",       "储蓄属性",       "E. 财务/合同", True,  "重要资料"),
    ("dividend_type",      "分红类型",       "E. 财务/合同", True,  "红利理念"),
    ("cash_value",         "保证现金价值",   "E. 财务/合同", True,  "重要资料"),
    ("inflation_resist",   "抗通胀能力",     "E. 财务/合同", True,  "产品特点"),
    ("premium_stability",  "保费稳定性",     "E. 财务/合同", False, "产品资料"),
    ("guaranteed_renew",   "保证续保",       "E. 财务/合同", False, "保障条款"),
    ("death_benefit",      "身故赔偿",       "E. 财务/合同", False, "产品资料"),
    ("plan_currency",      "保单货币",       "E. 财务/合同", False, "产品资料"),
]


# =============================================================================
# 数据文件路径
# =============================================================================
# 预制文件：代码附带的完整数据（值 + 引用），永远存在，作为兜底
PRESET_DATA_PATH    = DATA_DIR / "products_preset.json"
# 提取文件：PDF 提取后落地，优先使用；不存在时降级到预制文件
EXTRACTED_DATA_PATH = DATA_DIR / "products_extracted.json"


def _load_product_data() -> tuple[dict, dict]:
    """
    加载产品数据。
    优先读取 products_extracted.json（PDF 提取的真实数据），
    不存在则读取 products_preset.json（预制数据）。

    返回：
        data:      {field_key: {product_code: value_str}}   → 供 TableBuildSkill 使用
        citations: {field_key: {product_code: citation_str}} → 供引用列显示
    """
    # 决定读哪个文件
    if EXTRACTED_DATA_PATH.exists():
        src_path = EXTRACTED_DATA_PATH
        src_name = "products_extracted.json (PDF提取)"
    else:
        src_path = PRESET_DATA_PATH
        src_name = "products_preset.json (预制数据)"

    with open(src_path, encoding="utf-8") as f:
        raw = json.load(f)   # {code: {field: {value, citation}}}

    print(f"[数据加载] 使用 {src_name}")

    # 转换格式
    data:      dict[str, dict[str, str]] = {}  # {field: {code: value}}
    citations: dict[str, dict[str, str]] = {}  # {field: {code: citation}}

    for code, fields in raw.items():
        for fk, entry in fields.items():
            if isinstance(entry, dict):
                val  = entry.get("value",    "N/A")
                cite = entry.get("citation", "")
            else:
                # 兼容旧格式（只有字符串值）
                val  = str(entry)
                cite = ""
            data.setdefault(fk, {})[code]      = val
            citations.setdefault(fk, {})[code] = cite

    return data, citations


# 模块级加载（启动时执行一次）
_PRODUCT_DATA, _CITATIONS = _load_product_data()


class TableBuildSkill:
    name = "table_build"

    @staticmethod
    def reload():
        """重新从文件加载数据（PDF 提取完成后调用）"""
        global _PRODUCT_DATA, _CITATIONS
        _PRODUCT_DATA, _CITATIONS = _load_product_data()
    def run(self, products: dict, extra_dims: list = None) -> dict:
        """
        从文件加载的数据（_PRODUCT_DATA / _CITATIONS）构建对比表。
        数据来源优先级：products_extracted.json > products_preset.json
        """
        codes = list(products.keys())
        rows  = []

        for (fk, label, group, primary, cite_hint) in DIMENSIONS:
            # 从模块级变量读取（文件驱动，无硬编码）
            field_data  = _PRODUCT_DATA.get(fk, {})
            values      = {c: field_data.get(c, "N/A") for c in codes}
            is_diff     = len(set(values.values())) > 1
            field_cites = _CITATIONS.get(fk, {})
            citation    = {c: field_cites.get(c, cite_hint) for c in codes}
            rows.append({
                "field":    fk,
                "label":    label,
                "group":    group,
                "primary":  primary,
                "values":   values,
                "citation": citation,
                "is_diff":  is_diff,
                "dynamic":  False,
            })

        # AI 动态追加的维度
        if extra_dims:
            for dim in extra_dims:
                values = {c: dim["values"].get(c, "N/A") for c in codes}
                rows.append({
                    "field":    f"dyn_{dim['name']}",
                    "label":    dim["name"],
                    "group":    "F. AI动态分析",
                    "primary":  True,
                    "values":   values,
                    "citation": {c: dim.get("citation", "AI分析") for c in codes},
                    "is_diff":  len(set(values.values())) > 1,
                    "dynamic":  True,
                })
        return {"rows": rows, "codes": codes}


class DiffDetectSkill:
    name = "diff_detect"
    def run(self, rows: list) -> dict:
        diffs = [r for r in rows if r["is_diff"]]
        dyns  = [r for r in rows if r.get("dynamic")]
        groups: dict = {}
        for r in diffs:
            groups.setdefault(r["group"], []).append(r["label"])
        return {
            "total_dims":    len(rows),
            "diff_count":    len(diffs),
            "dyn_count":     len(dyns),
            "diff_by_group": groups,
        }


# =============================================================================
# AGENT LOG
# =============================================================================

@dataclass
class AgentLog:
    agent:     str
    action:    str
    detail:    str
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# AGENTS
# =============================================================================

class PDFAgent:
    name = "PDF解析Agent"
    def __init__(self, col):
        self.parse_sk = PDFParseSkill()
        self.chunk_sk = SemanticChunkSkill()
        self.index_sk = VectorIndexSkill(col)
        self.logs: list = []
    def _log(self, a, d): self.logs.append(AgentLog(self.name, a, d))
    def ingest(self, code, pdf_path, product_name):
        self._log("skill:pdf_parse",      f"解析 {code}")
        r1 = self.parse_sk.run(pdf_path=pdf_path)
        self._log("skill:semantic_chunk", f"分块 {r1['total_pages']} 页")
        r2 = self.chunk_sk.run(r1["pages"], code, product_name)
        self._log("skill:vector_index",   f"写入 {r2['count']} chunks")
        r3 = self.index_sk.run(r2["chunks"], code)
        self._log("完成", f"pages={r1['total_pages']} indexed={r3['indexed']}")
        return r3


class RAGAgent:
    name = "RAG检索Agent"
    def __init__(self, col):
        self.retrieval_sk = RetrievalSkill(col)
        self.logs: list = []
    def _log(self, a, d): self.logs.append(AgentLog(self.name, a, d))
    def retrieve(self, query, product_codes, n=6):
        self._log("skill:retrieval", f"「{query[:35]}」→ {product_codes}")
        r = self.retrieval_sk.run(query, product_codes, n)
        chunks = r.get("chunks", [])
        self._log("结果", f"{len(chunks)} 个段落")
        return chunks


class CompareAgent:
    name = "对比分析Agent"
    def __init__(self):
        self.table_sk = TableBuildSkill()
        self.diff_sk  = DiffDetectSkill()
        self.logs: list = []
    def _log(self, a, d): self.logs.append(AgentLog(self.name, a, d))
    def compare(self, products, extra_dims=None):
        self._log("skill:table_build", f"产品：{list(products.keys())}")
        result = self.table_sk.run(products, extra_dims)
        self._log("skill:diff_detect", "差异分析")
        stats = self.diff_sk.run(result["rows"])
        self._log("完成", f"{stats['total_dims']} 维度，差异 {stats['diff_count']} 个")
        return {**result, "stats": stats}


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class Orchestrator:
    name = "Orchestrator"

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._col    = self._chroma.get_or_create_collection(
            "idss_products", metadata={"hnsw:space": "cosine"})

        self.pdf_agent     = PDFAgent(self._col)
        self.rag_agent     = RAGAgent(self._col)
        self.compare_agent = CompareAgent()

        self.selected_codes: list = []
        self.dynamic_dims:   list = []
        self.chat_history:   list = []
        self.own_logs:       list = []

        with open(DATA_DIR / "products_index.json", encoding="utf-8") as f:
            self.product_index: dict = json.load(f)

        self._indexed: set = set()
        self._lock = threading.Lock()
        self._restore_index()

    def _log(self, a, d): self.own_logs.append(AgentLog(self.name, a, d))

    def _restore_index(self):
        for code in PRODUCT_PATHS:
            try:
                r = self._col.get(where={"product_code": code}, limit=1)
                if r["ids"]: self._indexed.add(code)
            except Exception: pass
        if self._indexed:
            self._log("恢复状态", f"已索引：{sorted(self._indexed)}")

    def get_all_logs(self) -> list:
        all_logs = (self.own_logs + self.pdf_agent.logs
                    + self.rag_agent.logs + self.compare_agent.logs)
        return sorted(all_logs, key=lambda x: x.timestamp)

    def ensure_indexed(self, code, on_progress=None) -> bool:
        if code in self._indexed:
            return True
        with self._lock:
            if code in self._indexed:
                return True
            path = PRODUCT_PATHS.get(code)
            # PDF 不存在时优雅降级：跳过索引，对比表仍可用结构化数据生成
            if not path or not Path(path).exists():
                self._log("跳过索引", f"{code}: PDF未找到，将使用内置结构化数据")
                if on_progress:
                    on_progress(f"⚠️ {code} PDF未找到，使用内置数据")
                return False
            self._log("路由→PDF解析Agent", f"索引 {code}")
            if on_progress:
                on_progress(f"📄 正在索引 {code}…")
            name = PRODUCT_NAMES_ZH.get(code, code)
            try:
                self.pdf_agent.ingest(code, path, name)
                self._indexed.add(code)
                self._log("索引完成", f"{code} 就绪")
            except Exception as e:
                self._log("索引失败", f"{code}: {str(e)[:60]}，使用内置数据")
                if on_progress:
                    on_progress(f"⚠️ {code} 索引失败，使用内置数据")
            return True

    def set_selected_products(self, codes):
        self.selected_codes = codes
        self._log("状态更新", f"选择产品：{codes}")

    def get_comparison(self):
        if len(self.selected_codes) < 2: return None
        self._log("路由→对比分析Agent", f"{self.selected_codes}")
        products = {c: self.product_index[c]
                    for c in self.selected_codes if c in self.product_index}
        return self.compare_agent.compare(products, extra_dims=self.dynamic_dims)

    def chat(self, message: str) -> dict:
        from mock_llm import call_llm

        self._log("收到消息", message[:60])
        self.chat_history.append({"role": "user", "content": message})

        self._log("路由→RAG检索Agent", "检索相关段落")
        chunks = self.rag_agent.retrieve(message, self.selected_codes)

        self._log("路由→LLM", "生成回答")
        structured = {c: self.product_index.get(c, {}) for c in self.selected_codes}
        result = call_llm(
            query=message,
            selected_products=self.selected_codes,
            retrieved_chunks=chunks,
            chat_history=self.chat_history[-10:],  # 防止超token
            structured_data=structured,
        )

        answer  = result.get("answer", "")
        new_dim = result.get("new_dimension")
        backend = result.get("backend", "?")

        self.chat_history.append({"role": "assistant", "content": answer})
        self._log("回答完成", f"[{backend}] 长度 {len(answer)} 字，段落 {len(chunks)} 个")

        if new_dim and new_dim["name"] not in {d["name"] for d in self.dynamic_dims}:
            self.dynamic_dims.append(new_dim)
            self._log("动态维度", f"新增：{new_dim['name']}")

        return {
            "answer":        answer,
            "new_dimension": new_dim,
            "sources":       result.get("sources", []),
            "chunks_used":   len(chunks),
            "backend":       backend,
        }

    def rebuild_index_from_pdfs(self, on_progress=None) -> dict:
        """
        用 LLM 从 PDF 提取结构化数据，落地到 products_extracted.json。

        数据流：
          PDF → pdfplumber → LLM提取 → products_extracted.json（磁盘）
                                      → TableBuildSkill.reload()（热更新内存）

        生成对比表时自动优先读取 products_extracted.json，
        不存在则降级到 products_preset.json。

        仅在 Kimi/Ollama 模式下执行；Mock 模式返回错误提示。
        """
        from mock_llm import LLMConfig, LLMBackend, extract_product_data

        if LLMConfig.backend == LLMBackend.MOCK:
            msg = "Mock 模式下无法从 PDF 提取数据，请先切换到 Kimi 或 Ollama。"
            self._log("PDF提取", msg)
            if on_progress: on_progress(f"❌ {msg}")
            return {"error": msg}

        report = {}

        # 先加载已有的 extracted 文件（如果存在），用于增量更新
        if EXTRACTED_DATA_PATH.exists():
            try:
                with open(EXTRACTED_DATA_PATH, encoding="utf-8") as f:
                    all_extracted = json.load(f)
            except Exception:
                all_extracted = {}
        else:
            # 从预制文件初始化，确保未提取的产品也有数据
            with open(PRESET_DATA_PATH, encoding="utf-8") as f:
                all_extracted = json.load(f)

        for code, path in PRODUCT_PATHS.items():
            if not Path(path).exists():
                self._log("PDF提取", f"{code}: PDF不存在，跳过")
                if on_progress: on_progress(f"⚠️ {code}: PDF 未找到，跳过")
                report[code] = {"status": "skipped", "reason": "PDF not found"}
                continue

            name = PRODUCT_NAMES_ZH.get(code, code)
            self._log("PDF提取开始", f"{code} ({name})")
            if on_progress: on_progress(f"📄 开始处理 {name} ({code})…")

            # 1. 解析 PDF 页面
            try:
                parse_result = self.pdf_agent.parse_sk.run(pdf_path=str(path))
                pages = parse_result["pages"]
                self._log("PDF提取", f"{code}: 解析 {len(pages)} 页")
            except Exception as e:
                self._log("PDF提取失败", f"{code}: {str(e)[:80]}")
                if on_progress: on_progress(f"❌ {code} PDF 解析失败: {str(e)[:60]}")
                report[code] = {"status": "failed", "reason": str(e)}
                continue

            # 2. LLM 提取结构化数据
            try:
                extracted = extract_product_data(
                    product_code=code,
                    product_name=name,
                    pages=pages,
                    on_progress=on_progress,
                )
            except Exception as e:
                self._log("LLM提取失败", f"{code}: {str(e)[:80]}")
                if on_progress: on_progress(f"❌ {code} LLM 提取失败: {str(e)[:60]}")
                report[code] = {"status": "failed", "reason": str(e)}
                continue

            if not extracted:
                self._log("PDF提取", f"{code}: 未提取到任何字段")
                if on_progress: on_progress(f"⚠️ {code}: 未提取到数据，保留原有")
                report[code] = {"status": "empty"}
                continue

            # 3. 合并到 all_extracted
            # 格式统一为 {field: {value, citation}}
            # 以预制文件为底，用 LLM 提取的结果覆盖
            prod_data = all_extracted.get(code, {})
            for fk, entry in extracted.items():
                if isinstance(entry, dict) and "value" in entry:
                    prod_data[fk] = entry   # 完整保留 {value, citation}
            all_extracted[code] = prod_data

            n_extracted = len(extracted)
            n_total     = len(DIMENSIONS)
            self._log("PDF提取完成", f"{code}: {n_extracted}/{n_total} 字段")
            if on_progress:
                on_progress(f"✅ {name}: 提取 {n_extracted}/{n_total} 个字段")
            report[code] = {
                "status":           "ok",
                "fields_extracted": n_extracted,
                "fields_total":     n_total,
            }

        # 4. 落地到 products_extracted.json（完整保留 value + citation）
        try:
            with open(EXTRACTED_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(all_extracted, f, ensure_ascii=False, indent=2)
            self._log("PDF提取", "products_extracted.json 已保存")
            if on_progress: on_progress("💾 products_extracted.json 已保存到磁盘")
        except Exception as e:
            self._log("写入失败", str(e)[:80])
            if on_progress: on_progress(f"❌ 写入文件失败: {str(e)[:60]}")
            return report

        # 5. 热更新内存（无需重启，对比表立即使用新数据）
        TableBuildSkill.reload()
        self._log("PDF提取", "内存数据已热更新")
        if on_progress: on_progress("🔄 内存数据已热更新，对比表将使用 PDF 提取数据")

        return report

    def clear_session(self):
        self.selected_codes = []
        self.dynamic_dims   = []
        self.chat_history   = []
        self.own_logs       = []
        for a in (self.pdf_agent, self.rag_agent, self.compare_agent):
            a.logs.clear()
        self._log("会话重置", "已清除对话状态")
