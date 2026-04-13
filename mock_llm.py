"""
支持四种后端：
  1. mock     — 本地知识库，无需 API（默认演示模式）
  2. kimi     — Moonshot AI 云端 API（moonshot-v1-8k / 32k / 128k）
  3. deepseek — DeepSeek API（deepseek-chat / deepseek-reasoner）
  4. ollama   — 本地 Ollama

用法：
  from mock_llm import call_llm, LLMConfig, LLMBackend
  LLMConfig.set_backend(LLMBackend.DEEPSEEK, api_key="sk-xxx")
"""

import os
import json
import enum
import time
import re
import urllib.request
import urllib.error
from typing import Optional

# requests 库绕过 Windows SChannel 的 TLS renegotiation 问题
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
# 后端枚举 & 全局配置
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(enum.Enum):
    MOCK     = "mock"
    KIMI     = "kimi"
    DEEPSEEK = "deepseek"
    OLLAMA   = "ollama"


class LLMConfig:
    backend:    LLMBackend = LLMBackend.MOCK
    api_key:    Optional[str] = os.getenv("MOONSHOT_API_KEY")
    deepseek_api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    model:      Optional[str] = None
    ollama_url: str = "http://192.168.200.54:11434"

    # 环境变量自动检测优先级：DeepSeek > Kimi
    if deepseek_api_key:
        backend = LLMBackend.DEEPSEEK
        api_key = deepseek_api_key
    elif api_key:
        backend = LLMBackend.KIMI

    @classmethod
    def set_backend(cls, backend: LLMBackend,
                    model: str = None,
                    api_key: str = None,
                    ollama_url: str = None):
        cls.backend = backend
        if model:      cls.model      = model
        if api_key:    cls.api_key    = api_key
        if ollama_url: cls.ollama_url = ollama_url

    @classmethod
    def get_effective_model(cls) -> str:
        if cls.model:
            return cls.model
        defaults = {
            LLMBackend.MOCK:     "mock-local",
            LLMBackend.KIMI:     "moonshot-v1-8k",
            LLMBackend.DEEPSEEK: "deepseek-chat",
            LLMBackend.OLLAMA:   "qwen3:30b",
        }
        return defaults[cls.backend]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP 工具（不依赖 openai SDK，避免版本冲突）
# ─────────────────────────────────────────────────────────────────────────────

def _post_json(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    """
    HTTP POST，优先用 requests（绕过 Windows SChannel TLS renegotiation），
    fallback 到 http.client。带指数退避重试（3 次）。
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)

    last_err = None
    for attempt in range(3):
        try:
            if _HAS_REQUESTS:
                # requests 使用 OpenSSL，不受 Windows SChannel renegotiation 影响
                resp = _requests.post(
                    url, data=data, headers=req_headers,
                    timeout=timeout, verify=True
                )
                if resp.status_code in (429, 502, 503) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return resp.json()
            else:
                # fallback: http.client 直连
                import http.client, ssl, urllib.parse
                parsed  = urllib.parse.urlparse(url)
                host    = parsed.netloc
                path    = parsed.path + (f"?{parsed.query}" if parsed.query else "")
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode    = ssl.CERT_NONE
                conn = http.client.HTTPSConnection(host, timeout=timeout, context=ssl_ctx)
                try:
                    conn.request("POST", path, body=data,
                                 headers={**req_headers, "Connection": "close"})
                    r = conn.getresponse()
                    if r.status in (429, 502, 503) and attempt < 2:
                        r.read(); time.sleep(2 ** attempt); continue
                    chunks = []
                    while True:
                        chunk = r.read(65536)
                        if not chunk: break
                        chunks.append(chunk)
                    raw = b"".join(chunks).decode("utf-8")
                    if r.status >= 400:
                        raise RuntimeError(f"HTTP {r.status}: {raw[:200]}")
                    return json.loads(raw)
                finally:
                    try: conn.close()
                    except Exception: pass

        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"网络错误（{attempt+1}次重试后）: {type(e).__name__}: {e}"
            ) from e


def _call_kimi(messages: list, model: str, api_key: str,
               temperature: float = 0.3, max_tokens: int = 2000) -> str:
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    resp = _post_json(url, headers, body, timeout=60)
    return resp["choices"][0]["message"]["content"]


def _call_deepseek(messages: list, model: str, api_key: str,
                   temperature: float = 0.3, max_tokens: int = 2000,
                   timeout: int = 120, thinking: bool = False) -> str:
    """
    DeepSeek API — 使用 streaming 模式逐块读取，
    避免大响应体在 chunked transfer 中途断连。
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      True,   # 流式返回，边生成边读
    }
    if not thinking:
        body["thinking"] = {"type": "disabled", "budget_tokens": 0}

    last_err = None
    for attempt in range(3):
        try:
            if _HAS_REQUESTS:
                resp = _requests.post(
                    url, json=body, headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout, stream=True, verify=True
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                # 逐行读取 SSE 流
                result = []
                for line in resp.iter_lines(chunk_size=256):
                    if not line:
                        continue
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data:"):
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            delta = json.loads(chunk)
                            content = (delta.get("choices", [{}])[0]
                                           .get("delta", {})
                                           .get("content", ""))
                            if content:
                                result.append(content)
                        except json.JSONDecodeError:
                            pass
                return "".join(result)
            else:
                # fallback: 非流式 http.client
                body["stream"] = False
                return _post_json(url, headers, body, timeout=timeout)["choices"][0]["message"]["content"]

        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"DeepSeek调用失败（{attempt+1}次重试）: {type(e).__name__}: {e}"
            ) from e


def _call_ollama(messages: list, model: str, base_url: str,
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    body = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = _post_json(url, headers, body, timeout=60)
    return resp["message"]["content"]


def probe_ollama(base_url: str) -> dict:
    """探测 Ollama 是否可用，返回 {ok, models, error}"""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# PDF 结构化提取（专用函数，独立于问答流程）
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """你是一位专业的香港危疾保险产品分析师。
你的任务是从产品小册子的 PDF 文本中，精确提取结构化数据。
严格按照要求的 JSON 格式输出，不要输出任何其他内容。
所有 citation 必须包含真实页码（如 "第4页" 或 "p.4"）。"""

_EXTRACT_FIELDS = {
    "product_position":  "产品定位（如：全能型多次赔、基础型单次赔）",
    "plan_type":         "计划类型（如：终身分红保险、定期保险）",
    "coverage_period":   "保障年期（如：终身、至100岁、至80岁）",
    "premium_terms":     "缴费期（如：10年、18年、25年）",
    "issue_age":         "投保年龄范围（如：0-65岁）",
    "disease_total":     "受保疾病总数（如：115种）",
    "disease_severe":    "严重危疾种数（如：57种）",
    "disease_early":     "早期危疾种数（如：44种，或：不涵盖）",
    "disease_children":  "严重儿童疾病种数（如：13种，或：不涵盖）",
    "early_not_erode":   "早期危疾赔付是否不侵蚀主保额（是/否）",
    "payout_structure":  "赔偿结构类型（如：多次赔、单次赔、分组多次赔）",
    "max_payout_pct":    "最高总赔偿额占保额百分比（如：1100%）",
    "cancer_claims":     "癌症最多可赔偿次数（如：6次、1次）",
    "heart_claims":      "心脏病最多可赔偿次数",
    "stroke_claims":     "中风最多可赔偿次数",
    "other_multiple":    "其他疾病是否支持多重赔偿（是/否及说明）",
    "cancer_trigger":    "癌症再次赔偿触发条件（如：复发、新发、持续）",
    "cancer_interval":   "癌症两次赔偿之间的间隔期要求",
    "cancer_cashflow":   "是否有癌症持续现金流赔偿（如：每月赔付最长100个月）",
    "first_ci_payout":   "首次严重危疾赔付比例（通常为100%保额）",
    "early_payout_pct":  "早期危疾预支比例及上限金额",
    "waiting_period":    "等候期（如：90天，癌症3年）",
    "survival_period":   "生存期要求（如：15天）",
    "icu_coverage":      "是否有ICU重症监护保障",
    "congenital":        "是否保障先天性疾病",
    "benign_tumor":      "是否保障良性肿瘤/病变",
    "dementia_annuity":  "是否有脑退化症/认知障碍年金保障",
    "diabetes_coverage": "糖尿病及相关慢性病保障情况",
    "pregnancy_cover":   "孕期及儿童特有疾病保障",
    "premium_waiver":    "保费豁免条款（在何种情况下豁免）",
    "sa_increase":       "保额提升机制（如：黄金岁月升级、Top-up额外保费）",
    "savings_attr":      "储蓄属性强弱（强/中/弱/无）",
    "dividend_type":     "分红类型（如：年度红利+终期分红、仅终期分红、无）",
    "cash_value":        "是否有保证现金价值",
    "inflation_resist":  "抗通胀能力（强/中/弱）及依据",
    "premium_stability": "保费稳定性（如：缴费期内固定、续保时按年龄递增）",
    "guaranteed_renew":  "是否保证续保",
    "death_benefit":     "身故赔偿内容（如：保额+分红、保额+双重分红）",
    "plan_currency":     "保单货币（如：港元/美元）",
}


def extract_product_data(
    product_code: str,
    product_name: str,
    pages: dict,          # {page_num: text}
    on_progress=None,     # callback(msg: str)
) -> dict:
    """
    用 LLM 从 PDF 页面文本中提取结构化产品数据。
    返回: {
        field_key: {"value": "...", "citation": "§章节 (p.X)"},
        ...
    }
    仅在 Kimi 或 Ollama 模式下有意义；Mock 模式直接返回空字典。
    """
    backend = LLMConfig.backend
    if backend == LLMBackend.MOCK:
        return {}

    model      = LLMConfig.get_effective_model()
    api_key    = LLMConfig.api_key
    ollama_url = LLMConfig.ollama_url

    # 提取时自动升级到更大模型，避免超 token
    extract_model = model
    if backend == LLMBackend.KIMI and "8k" in model:
        extract_model = "moonshot-v1-32k"
    elif backend == LLMBackend.DEEPSEEK and model == "deepseek-chat":
        extract_model = "deepseek-chat"   # DeepSeek-V3 context 64k，无需升级

    def _call(messages):
        if backend == LLMBackend.KIMI:
            return _call_kimi(messages, extract_model, api_key,
                              temperature=0.0, max_tokens=3000)
        elif backend == LLMBackend.DEEPSEEK:
            # DeepSeek 响应体较大，用更长的 timeout 防止 IncompleteRead
            return _call_deepseek(messages, extract_model, api_key,
                                  temperature=0.0, max_tokens=3000, timeout=120)
        else:
            return _call_ollama(messages, model, ollama_url,
                                temperature=0.0, max_tokens=3000)

    # ── 分批：每批 3 页，每页截断到 800 字，避免超 token ─────────────────────
    page_items = sorted(pages.items())          # [(page_num, text), ...]
    batch_size  = 3
    batches     = [page_items[i:i+batch_size]
                   for i in range(0, len(page_items), batch_size)]

    # 字段说明列表
    fields_desc = "\n".join(
        f'  "{k}": {v}'
        for k, v in _EXTRACT_FIELDS.items()
    )

    merged: dict = {}   # field_key -> {value, citation}

    for b_idx, batch in enumerate(batches):
        if on_progress:
            on_progress(f"[{product_code}] 提取第 {b_idx+1}/{len(batches)} 批页面…")

        # 拼接本批文本，每页最多 800 字（控制 token 用量）
        batch_text = ""
        for pg, txt in batch:
            if txt.strip():
                batch_text += f"\n\n【第{pg}页】\n{txt[:800]}"

        if not batch_text.strip():
            continue

        prompt = f"""产品名称：{product_name}（代码：{product_code}）

请从以下 PDF 页面文本中，尽可能提取下列字段的信息。
如果某字段在本批文本中找不到依据，直接省略该字段（不要猜测）。

需要提取的字段：
{fields_desc}

输出格式（严格 JSON，不要加任何说明文字）：
{{
  "字段key": {{
    "value": "提取到的具体值",
    "citation": "来源描述，格式如：§保障疾病赔偿一览表 (p.4) 或 第4页第3条"
  }},
  ...
}}

PDF 文本内容：
{batch_text}"""

        try:
            raw = _call([
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": prompt},
            ])

            # 提取 JSON（去掉可能的 markdown 代码块）
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r'\s*```$',          '', raw.strip(), flags=re.MULTILINE)

            batch_result = json.loads(raw)
            if isinstance(batch_result, dict):
                # 合并：先出现的页面优先（前面批次的结果不被后面覆盖）
                for k, v in batch_result.items():
                    if k not in merged and isinstance(v, dict) and "value" in v:
                        merged[k] = v

        except json.JSONDecodeError:
            # 尝试用正则从非标准输出中抢救
            try:
                m = re.search(r'\{[\s\S]+\}', raw)
                if m:
                    batch_result = json.loads(m.group())
                    for k, v in batch_result.items():
                        if k not in merged and isinstance(v, dict) and "value" in v:
                            merged[k] = v
            except Exception:
                pass
        except Exception as e:
            if on_progress:
                on_progress(f"[{product_code}] 批次{b_idx+1} 解析失败: {str(e)[:60]}")

    if on_progress:
        on_progress(f"[{product_code}] 提取完成，共 {len(merged)} 个字段 ✓")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Mock 后端：基于结构化知识库直接回答，无需 API
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_KB = {
    "癌症": {
        "answer": (
            "**癌症保障对比**\n\n"
            "- **二号产品 (IDSS2)**：最多赔偿 **6次**，支持复发/新发/持续三种触发，"
            "且提供每月现金流最长100个月。引用来源：IDSS2 产品小册子第4页\n"
            "- **五号产品 (IDSS5)**：最多赔偿 3次，间隔期约3年，无持续现金流。"
            "引用来源：IDSS5 产品小册子第5页\n"
            "- **一号产品 (IDSS1)**：2次（Big3守护），仅限复发触发。"
            "引用来源：IDSS1 产品小册子第4页\n"
            "- **三号产品 (IDSS3)** / **四号产品 (IDSS4)**：各1次。\n\n"
            "**结论**：癌症保障最强为 IDSS2，多次赔偿且有现金流设计。"
        ),
        "dim": {"name": "癌症最大赔偿次数", "values": {
            "IDSS1": "2次", "IDSS2": "6次", "IDSS3": "1次", "IDSS4": "1次", "IDSS5": "3次"
        }},
    },
    "心脏": {
        "answer": (
            "**心脏病保障对比**\n\n"
            "- **IDSS2**：心脏病最多赔偿2次（与中风合计3次上限）。来源：IDSS2 第4页\n"
            "- **IDSS1**：心脏病各多赔1次（Big3守护）。来源：IDSS1 第4页\n"
            "- **IDSS5**：心脏病最多2次。来源：IDSS5 第5页\n"
            "- **IDSS3 / IDSS4**：各1次。"
        ),
        "dim": None,
    },
    "icu": {
        "answer": (
            "**ICU 保障对比**\n\n"
            "在5款产品中，**仅二号产品(IDSS2) 提供ICU保障**，其他4款均不涵盖此项。"
            "来源：IDSS2 产品小册子特色保障章节"
        ),
        "dim": {"name": "ICU保障", "values": {
            "IDSS1": "❌ 无", "IDSS2": "✅ 有（唯一）", "IDSS3": "❌ 无",
            "IDSS4": "❌ 无", "IDSS5": "❌ 无"
        }},
    },
    "储蓄": {
        "answer": (
            "**储蓄分红对比**\n\n"
            "- **IDSS3 / IDSS5**：最强，提供年度红利+终期分红双层分红机制，现金价值稳健增长。"
            "来源：IDSS3 重要资料；IDSS5 产品小册子\n"
            "- **IDSS1 / IDSS2**：中等，仅终期分红（5年后启动）。\n"
            "- **IDSS4**：纯保障型，无储蓄成分。\n\n"
            "**结论**：以储蓄增值为目标，IDSS3 或 IDSS5 更适合。"
        ),
        "dim": None,
    },
    "先天": {
        "answer": (
            "**先天疾病保障对比**\n\n"
            "- **IDSS5（五号产品）**：市场首创，投保后发现的先天疾病亦可保障。来源：IDSS5 产品小册子特色章节\n"
            "- **IDSS2**：提供先天疾病保障（含儿童疾病13种）。来源：IDSS2 第3页\n"
            "- **IDSS1 / IDSS3 / IDSS4**：不涵盖先天疾病。"
        ),
        "dim": None,
    },
    "多重": {
        "answer": (
            "**多重赔偿对比**\n\n"
            "- **IDSS2**：不完全分组多次赔，灵活度最高，总赔偿可达保额1100%。来源：IDSS2 第4页\n"
            "- **IDSS5**：分组多次赔，总赔偿可达900%，含脑退化等分组。来源：IDSS5 第5页\n"
            "- **IDSS1**：仅限三大疾病（Big3）各多赔1次，总约280%。\n"
            "- **IDSS3 / IDSS4**：单次赔，100%。"
        ),
        "dim": None,
    },
    "儿童": {
        "answer": (
            "**儿童疾病保障对比**\n\n"
            "- **IDSS2 / IDSS4 / IDSS5**：覆盖严重儿童疾病13种（含先天性心脏病等）。来源：产品小册子保障疾病一览表\n"
            "- **IDSS3**：覆盖7种儿童疾病。\n"
            "- **IDSS1**：不涵盖儿童疾病。\n\n"
            "**结论**：为儿童投保，IDSS2 保障最全面。"
        ),
        "dim": {"name": "严重儿童疾病种数", "values": {
            "IDSS1": "❌ 0种", "IDSS2": "✅ 13种", "IDSS3": "✅ 7种",
            "IDSS4": "✅ 13种", "IDSS5": "✅ 13种"
        }},
    },
    "糖尿病": {
        "answer": (
            "**糖尿病 / 慢性病保障对比**\n\n"
            "- **IDSS5（五号产品）**：保障最全面，早期→严重阶段均涵盖。来源：IDSS5 保障疾病一览表\n"
            "- **IDSS3 / IDSS4**：中等，涵盖视网膜病变+血管介入手术。\n"
            "- **IDSS2**：基础，仅视网膜病变。\n"
            "- **IDSS1**：不涵盖。\n\n"
            "**结论**：有糖尿病风险首选 IDSS5。"
        ),
        "dim": None,
    },
}

def _mock_answer(query: str, selected_products: list, structured_data: dict) -> dict:
    """基于关键词匹配返回预设答案，支持动态新增维度"""
    q_lower = query.lower()

    # 关键词匹配
    matched = None
    for kw, entry in _MOCK_KB.items():
        if kw in q_lower or kw in query:
            matched = entry
            break

    if matched:
        # 过滤维度值只显示已选产品
        dim = None
        if matched["dim"] and selected_products:
            filtered_vals = {k: v for k, v in matched["dim"]["values"].items()
                             if k in selected_products}
            if filtered_vals:
                dim = {**matched["dim"], "values": filtered_vals}
        return {
            "answer":        matched["answer"],
            "new_dimension": dim,
            "sources":       ["产品小册子（结构化知识库）"],
            "backend":       "mock",
        }

    # 通用回答
    prods = "、".join(selected_products) if selected_products else "已选产品"
    return {
        "answer": (
            f"关于「{query}」，以下是已选产品（{prods}）的分析：\n\n"
            "目前该问题需结合具体保单条款判断。建议您：\n"
            "- 参阅各产品官方小册子的对应章节\n"
            "- 或联系寿险顾问获取专业建议\n\n"
            "如需对比某个具体维度（如癌症赔偿、ICU保障、储蓄分红等），"
            "请直接提问，我会为您详细比较。"
        ),
        "new_dimension": None,
        "sources":       ["产品结构化知识库"],
        "backend":       "mock",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kimi 后端：RAG + LLM 真实调用
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是寿险保险产品专业比较助手。用户正在比较以下危疾保险产品：{products}

你的职责：
1. 基于提供的产品知识，准确、客观地回答用户问题
2. 回答时引用来源（产品名称和页码），格式：「来源：产品名 第X页」
3. 如果用户问题涉及一个新的比较维度，在答案最后附加：
   [NEW_DIM]{{
     "name": "维度名称（简洁，10字以内）",
     "values": {{"IDSS1": "值", "IDSS2": "值", "IDSS3": "值", "IDSS4": "值", "IDSS5": "值"}},
     "citation": "来源章节"
   }}[/NEW_DIM]
   注意：只为已选产品填值，其他产品用"N/A"
4. 使用简体中文，语气专业但易懂
5. 不要编造数据，不确定的内容直接说明

产品知识库：
{knowledge}
"""

_PRODUCT_KNOWLEDGE = {
    "IDSS1": """一号产品（IDSS1）
- 类型：终身分红保险，保障58种危疾
- 癌症：最多2次（Big3守护），触发条件：复发
- 三大疾病（癌症/心脏/中风）各可多赔1次
- 无早期危疾保障
- 无ICU保障
- 黄金岁月：保额升级+40%
- 保费缴付：10/18/25年
- 身故赔偿：保额+分红
- 保单货币：HKD/USD
来源：一号产品小册子""",

    "IDSS2": """二号产品（IDSS2）
- 类型：终身分红保险，保障115种疾病（57严重+44早期+13儿童+1疾病组合）
- 癌症：最多6次，支持复发/新发/持续三种触发，提供每月癌症现金流（最长100个月）
- 心脏+中风合计最多3次，各类别最多2次
- 早期危疾不侵蚀主保额（市场优势）
- ✅ 唯一有ICU保障的产品
- ✅ 脑退化/柏金逊症年金
- ✅ 孕期及儿童保障（13种严重儿童疾病）
- ✅ 先天疾病保障
- 保费灵活Top-up
- 抗通胀能力强
来源：二号产品小册子""",

    "IDSS3": """三号产品（IDSS3）
- 类型：终身至100岁，保障100种疾病（52严重+39早期+7儿童）
- 早期危疾：会侵蚀主保额
- 单次赔偿，最高100%保额
- 储蓄属性强：年度红利+终期分红双层分红
- 保障升级选项：+50%
- 保费缴付：10/18/25年
来源：三号产品小册子""",

    "IDSS4": """四号产品（IDSS4）
- 类型：定期保险（5年续保至80岁），115种疾病
- 纯保障型，无储蓄成分，无现金价值
- 续保时保费按年龄递增
- 单次赔偿，最高100%保额
- 早期危疾10-20%（上限HKD 400,000）
- 无保费豁免
来源：四号产品小册子""",

    "IDSS5": """五号产品（IDSS5）
- 类型：终身分红保险，115种疾病
- 癌症：最多3次，间隔期约3年
- 分组多次赔，总赔偿最高900%
- ✅ 先天疾病保障（市场首创，投保后发现亦保）
- ✅ 良性病变保障（市场首创）
- 糖尿病保障最全面（早期→严重全阶段）
- 癌症等候期3年（较长）
- 年度红利+终期分红双层分红
- 子女保单保费豁免
来源：五号产品小册子""",
}


def _build_rag_context(selected_products: list, retrieved_chunks: list) -> str:
    """组装知识库上下文：结构化知识 + RAG检索片段"""
    ctx = ""
    # 1. 结构化知识
    for code in selected_products:
        if code in _PRODUCT_KNOWLEDGE:
            ctx += f"\n{'='*50}\n{_PRODUCT_KNOWLEDGE[code]}\n"
    # 2. RAG检索片段（如果有）
    if retrieved_chunks:
        ctx += f"\n{'='*50}\n【从PDF检索到的相关片段】\n"
        for i, chunk in enumerate(retrieved_chunks[:5]):
            src  = chunk.get("source", "未知来源")
            text = chunk.get("text",   "")[:400]
            ctx += f"[片段{i+1}] {src}\n{text}\n\n"
    return ctx


def _extract_new_dim(text: str) -> Optional[dict]:
    """从 LLM 回答中提取 [NEW_DIM]...[/NEW_DIM] 块"""
    m = re.search(r'\[NEW_DIM\](.*?)\[/NEW_DIM\]', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except Exception:
        return None


def _clean_answer(text: str) -> str:
    """移除 NEW_DIM 标记"""
    return re.sub(r'\[NEW_DIM\].*?\[/NEW_DIM\]', '', text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    query:             str,
    selected_products: list,
    retrieved_chunks:  list  = None,
    chat_history:      list  = None,
    structured_data:   dict  = None,
) -> dict:
    """
    统一调用入口，根据 LLMConfig.backend 自动路由。
    返回：{answer, new_dimension, sources, backend}
    """
    retrieved_chunks = retrieved_chunks or []
    chat_history     = chat_history     or []
    structured_data  = structured_data  or {}

    backend = LLMConfig.backend

    # ── Mock 模式 ──────────────────────────────────────────────────────────
    if backend == LLMBackend.MOCK:
        return _mock_answer(query, selected_products, structured_data)

    # ── Kimi / Ollama 真实模式 ─────────────────────────────────────────────
    model   = LLMConfig.get_effective_model()
    knowledge = _build_rag_context(selected_products, retrieved_chunks)
    prod_str  = "、".join(selected_products)

    system_msg = _SYSTEM_PROMPT.format(products=prod_str, knowledge=knowledge)

    messages = [{"role": "system", "content": system_msg}]
    # 加入最近 6 轮历史（避免超 token）
    for h in chat_history[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": query})

    try:
        if backend == LLMBackend.KIMI:
            raw = _call_kimi(
                messages, model, LLMConfig.api_key,
                temperature=0.3, max_tokens=2000
            )
        elif backend == LLMBackend.DEEPSEEK:
            raw = _call_deepseek(
                messages, model, LLMConfig.api_key,
                temperature=0.3, max_tokens=2000,
                timeout=60, thinking=False
            )
        else:  # OLLAMA
            raw = _call_ollama(
                messages, model, LLMConfig.ollama_url,
                temperature=0.3, max_tokens=2000
            )

        new_dim = _extract_new_dim(raw)
        answer  = _clean_answer(raw)

        # 若 LLM 没有给出新维度格式但检测到新维度意图，从 mock 补充
        if not new_dim:
            mock_result = _mock_answer(query, selected_products, structured_data)
            new_dim = mock_result.get("new_dimension")

        return {
            "answer":        answer,
            "new_dimension": new_dim,
            "sources":       [c.get("source", "") for c in retrieved_chunks[:3]],
            "backend":       backend.value,
        }

    except Exception as e:
        # 降级到 Mock
        result = _mock_answer(query, selected_products, structured_data)
        result["answer"] = f"⚠️ {backend.value} 调用失败（{e}），已降级到本地模式。\n\n" + result["answer"]
        result["backend"] = f"{backend.value}→mock(fallback)"
        return result
