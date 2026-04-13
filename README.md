# IDSS Advisor — 保险产品智能决策支持系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NiceGUI-1.4%2B-brightgreen" />
  <img src="https://img.shields.io/badge/ChromaDB-向量数据库-orange" />
  <img src="https://img.shields.io/badge/LLM-Kimi%20%7C%20Ollama%20%7C%20DeepSeek-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

一个基于 AI 的 Web 应用，支持对多款危疾保险产品进行**全维度对比**，并通过自然语言问答、基于 PDF 小册子的 RAG 检索、动态维度追加等功能，帮助用户做出更明智的购买决策——全程响应时间控制在 **10 秒以内**。

系统采用**多智能体（Multi-Agent）架构**：一个 Orchestrator 协调 PDF 解析 Agent、RAG 检索 Agent 和对比分析 Agent，每个 Agent 配备专属技能（Skills）。

![alt tag](https://raw.githubusercontent.com/houalexdev/idss-advisor/main/idss-advisor.png)

---

## 目录

- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [LLM 后端对比](#llm-后端对比)
- [配置参考](#配置参考)
- [扩展系统](#扩展系统)
- [常见问题](#常见问题)
- [开源协议](#开源协议)

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **PDF 驱动的真实数据提取** | 点击 **📄 从 PDF 重建数据**，LLM 自动读取每份产品小册子，提取全部 40 个维度并附带精确页码引用，替换内置静态数据 |
| **全维度对比表** | 40 个维度，分 5 大组：产品定位、疾病覆盖、赔偿机制、特色保障、财务/合同 |
| **差异高亮** | 各产品存在差异的字段用 ⚡ 标记 |
| **单元格引用** | 每个数据格均显示来源章节名和页码，数据可溯源 |
| **分组折叠/展开** | 点击分组标题（▶）展开扩展维度，再次点击收起 |
| **AI 问答面板** | 用自然语言询问任何产品相关问题 |
| **动态维度追加** | 问答回答中识别到新维度时，自动向对比表追加新行 |
| **8 个快速提问按钮** | 预设问题：癌症、心脏、儿童、糖尿病、储蓄、多重赔、ICU、先天疾病 |
| **PDF 原文链接** | 点击表头 **📄 查看原文 PDF** 在新标签页打开官方小册子 |
| **LLM 热切换** | 运行时切换 DeepSeek / Kimi / Ollama，无需重启 |
| **RAG 语义检索** | PDF 小册子经过分块、嵌入，存入 ChromaDB 进行语义搜索 |
| **Agent 运行日志** | 实时展示所有 Agent 动作和技能调用记录 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    NiceGUI Web UI  (Port 8080)                   │
│  ┌──────────────┐  ┌────────────────────────────┐  ┌───────────┐ │
│  │   Product    │  │      Comparison Table      │  │   Chat    │ │
│  │   Selector   │  │  (40 dims · up to 5 prods) │  │   Panel   │ │
│  └──────┬───────┘  └─────────────┬──────────────┘  └─────┬─────┘ │
└─────────┼────────────────────────┼───────────────────────┼───────┘
          │                        │                       │
          ▼                        ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Orchestrator Agent                       │
│         Routes requests · session state · chat history           │
│                                                                  │
│  ┌───────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │     PDF Agent     │  │  Compare Agent   │  │   RAG Agent   │  │
│  │  • pdf_parse      │  │  • table_build   │  │  • retrieval  │  │
│  │  • semantic_chunk │  │    (40 dims +    │  │   (ChromaDB   │  │
│  │  • vector_index   │  │     citations)   │  │    cosine)    │  │
│  └────────┬──────────┘  │  • diff_detect   │  └──────┬────────┘  │
│           │             └──────────────────┘         │           │
└───────────┼──────────────────────────────────────────┼───────────┘
            ▼                                          ▼
   ┌─────────────────┐                     ┌────────────────────┐
   │    ChromaDB     │◄────────────────────│    LLM Router      │
   │  Vector Store   │   RAG context       │  DeepSeek / Kimi / │
   │  (persistent)   │                     │  Ollama            │
   └─────────────────┘                     └────────────────────┘
```

### Agent 与 Skill 说明

| Agent | 技能（Skills） | 职责 |
|-------|---------------|------|
| **Orchestrator** | — | 路由所有请求；持有会话状态、对话历史（最近10轮）、动态维度列表 |
| **PDF解析Agent** | `pdf_parse`、`semantic_chunk`、`vector_index` | 解析 PDF（文本 + 表格），语义分块，写入 ChromaDB |
| **RAG检索Agent** | `retrieval` | 对 ChromaDB 做余弦相似度检索，每次返回最相关的 6 个段落 |
| **对比分析Agent** | `table_build`、`diff_detect` | 构建含精确来源引用的 40 维对比表，标记差异单元格 |

### 数据流

```
用户选择产品
        │
        ▼
Orchestrator.ensure_indexed(code)
        ├─ 有PDF？ ──► PDF Agent → pdfplumber → 分块 → ChromaDB
        └─ 无PDF？ ──► 降级使用 products_index.json（对比表仍可正常生成）
        │
        ▼
对比分析Agent.compare()
        ├─ table_build  → 40行对比表，每格含 CITATIONS 字典中的章节+页码
        └─ diff_detect  → 标记差异字段
        │
        ▼  （用户提问时）
RAG检索Agent.retrieve(query)     ← ChromaDB 语义检索
        │
        ▼
LLM路由器.call_llm()
        ├─ DeepSeek   → DeepSeek AI API
        ├─ Kimi       → Moonshot AI API，指数退避重试
        └─ Ollama     → 本地 /api/chat 接口
        │
        ▼
回答 + 可选的 new_dimension  →  动态追加到对比表
```

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web UI | [NiceGUI](https://nicegui.io) 1.4+ | Python 原生响应式 UI，无需编写前端代码 |
| 向量数据库 | [ChromaDB](https://www.trychroma.com) | 持久化存储，余弦相似度检索 |
| PDF 解析 | [pdfplumber](https://github.com/jsvine/pdfplumber) | 逐页提取文本和表格 |
| 云端 LLM | [DeepSeek AI / DeepSeek](https://platform.deepseek.com) | `deepseek-chat` |
| 云端 LLM | [Moonshot AI / Kimi](https://platform.moonshot.cn) | `moonshot-v1-8k / 32k / 128k` |
| 本地 LLM | [Ollama](https://ollama.com) | 推荐模型：`qwen3:30b` |
| HTTP 客户端 | Python `urllib` | 无需第三方 SDK |
| Python | 3.10+ | 使用内置泛型类型注解（`dict[str, str]`） |

---

## 项目结构

```
idss-advisor/
├── app.py                   # 主应用
│                            #   · 三栏响应式 NiceGUI 布局
│                            #   · 基于 asyncio 事件循环的线程安全 UI 更新
│                            #   · LLM 面板实时热切换
│                            #   · 全局 JS 事件委托处理 PDF 链接
│
├── agents.py                # 多智能体系统
│                            #   · Orchestrator、PDFAgent、RAGAgent、CompareAgent
│                            #   · CITATIONS 字典：40维度 × 5产品 × 章节+页码
│                            #   · TableBuildSkill._DATA：全产品数据
│                            #   · PDFParseSkill 含表格提取
│
├── mock_llm.py              # LLM 路由器
│                            #   · DeepSeek：urllib 直接调用，指数退避重试
│                            #   · Kimi：urllib 直接调用，指数退避重试
│                            #   · Ollama：/api/chat 接口
│                            #   · 启动时自动检测 MOONSHOT_API_KEY
│
├── requirements.txt         # Python 依赖
├── env.example              # 环境变量模板
├── LICENSE                  # MIT 开源协议
├── README.md                # 英文文档
├── README_zh.md             # 本文件（中文）
│
└── data/
    ├── products_index.json                       # 结构化数据索引（必需）
    ├── 01.pdf                                    # 一号产品       （可选）
    ├── 02.pdf                                    # 二号产品       （可选）
    ├── 03.pdf                                    # 三号产品       （可选）
    ├── 04.pdf                                    # 四号产品       （可选）
    └── 05.pdf                                    # 五号产品       （可选）

chroma_db/                   # ChromaDB 向量存储（首次运行时自动创建）
```

> **关于 PDF 文件**：PDF 为可选项。未放置 PDF 时，系统会自动降级使用 `products_index.json`，对比表**完全正常工作**。PDF 文件仅用于增强问答面板的 RAG 检索质量。

---

## 快速开始

### 环境要求

- Python **3.10 或以上**（推荐 3.11）
- pip

### 1. 克隆仓库

```bash
git clone https://github.com/houalexdev/idss-advisor.git
cd idss-advisor
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLM 后端（四选一）

**方案 A — DeepSeek / DeepSeek AI**（推荐，回答质量最佳）

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key：
DEEPSEEK_API_KEY=sk-your-key-here
```

申请地址：https://platform.deepseek.com  
启动时系统自动检测到 Key 并切换至 DeepSeek 模式。

**方案 B — Kimi / Moonshot AI**（推荐，回答质量最佳）

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key：
MOONSHOT_API_KEY=sk-your-key-here
```

申请地址：https://platform.moonshot.cn  
启动时系统自动检测到 Key 并切换至 Kimi 模式。

**方案 C — Ollama 本地模型**（离线 / 隐私优先）

```bash
# 安装 Ollama：https://ollama.com
ollama pull qwen3:30b
# 无需 .env，启动后在 UI 中配置地址和模型
```

**方案 D — Mock 模式**（零配置，即开即用）

无需任何配置，是没有设置 API Key 时的默认模式。  
内置知识库覆盖全部 8 个快速提问主题，响应几乎零延迟。

### 4. 放置 PDF 文件（可选）

从保险公司官网下载官方产品小册子，放入 `data/` 目录，文件名需与以下完全一致：

| 文件名 | 产品 |
|--------|------|
| `01.pdf` | 一号产品 (IDSS1) |
| `02.pdf` | 二号产品 (IDSS2) |
| `03.pdf` | 三号产品 (IDSS3) |
| `04.pdf` | 四号产品 (IDSS4) |
| `05.pdf` | 五号产品 (IDSS5) |

### 5. 启动

```bash
python app.py
```

用浏览器打开：**http://localhost:8080**

启动成功后控制台输出：

```
=======================================================
IDSS Insurance Decision Support System
LLM Backend : kimi (moonshot-v1-8k)
访问地址    : http://localhost:8080
=======================================================
```

---

## 使用指南

### PDF 驱动数据提取 *（核心功能 — 直接满足"PDF为权威数据源"要求）*

对比表数据可由 **LLM 直接从官方 PDF 小册子中提取**，而不仅仅依赖内置静态数据。

1. 将 5 份产品 PDF 放入 `data/` 目录（文件名见 [快速开始](#快速开始)）
2. 将 LLM 切换为 **DeepSeek** 或 **Kimi** 或 **Ollama**（需要 API Key 或本地安装）
3. 点击左栏系统状态卡片中的 **📄 从 PDF 重建数据**
4. 系统将执行：
   - 用 `pdfplumber` 逐页解析每份 PDF
   - 将页面文本分批喂给 LLM，使用 40 字段结构化提取提示词
   - 提取所有对比维度，附带**精确引用**（如 `§赔偿表 (p.4)`）
   - 热更新 `TableBuildSkill._DATA`、`CITATIONS` 和 `products_index.json`，**无需重启**
5. 正常点击 **② 生成对比分析 →**，所有数据均来自 LLM 对 PDF 的真实解析

> 未放置 PDF 或在 Mock 模式下，系统使用内置结构化数据集，对比表同样完整可用。

---

### 标准操作流程

1. **选择产品** — 在左栏勾选 2–5 款产品
2. **生成对比** — 点击 **"② 生成对比分析 →"**（勾选 ≥2 款后按钮激活）
3. **浏览对比表** — 上下滚动查看全部 40 维度；点击 **▶** 分组标题展开隐藏维度
4. **查看 PDF 原文** — 点击任意列表头中的 **📄 查看原文 PDF**（在新标签页打开）
5. **查看来源引用** — 每个数据格下方显示章节名和页码
6. **提问** — 在问答面板输入问题，或点击快速提问按钮
7. **观察表格动态更新** — AI 回答中识别到新维度时，自动在表格中追加新行
8. **切换 LLM** — 点击 LLM 面板中的任意选项，填写配置后点击 **应用**
9. **重置会话** — 点击 **🔄 重置会话** 清除全部状态

### 快速提问按钮

| 按钮 | 对应问题 |
|------|---------|
| 🎗 癌症 | 哪个计划癌症保障最好，可以多次赔偿吗？ |
| ❤️ 心脏 | 心脏病保障哪个最全面？ |
| 👶 儿童 | 哪个计划严重儿童疾病保障最好？ |
| 💊 糖尿病 | 如果我有糖尿病风险，应该选哪个计划？ |
| 💰 储蓄 | 哪个计划储蓄分红回报最好？ |
| 🔁 多重 | 多重赔偿哪个计划最强？ |
| 🏥 ICU | 哪个计划有 ICU 保障？ |
| 🧬 先天 | 先天疾病保障哪个最好？ |

---

## LLM 后端对比

| 模式 | 延迟 | 需要 | 适合场景 |
|------|------|------|---------|
| **Mock** | ~0 ms | 无需配置 | 演示、离线、零延迟测试 |
| **Kimi** `moonshot-v1-8k` | 2–8 s | Moonshot API Key | 生产级回答质量 |
| **DeepSeek** `deepseek-v3` | 1–5 s | DeepSeek API Key | 长文本、复杂多产品查询 |
| **Ollama** `qwen3:8b` | 2–8 s | 本地 Ollama 安装 | 更快的本地响应，GPU 占用低 |

**推荐本地模型**：`qwen3:8b` — 对中文危疾险文档的理解质量和响应速度最均衡。

所有后端均支持**自动降级**： DeepSeek / Kimi / Ollama 经过 3 次指数退避重试失败后，自动降级到 Mock 模式，并在回答前加入警告提示。

---

## 配置参考

### `.env` 文件

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MOONSHOT_API_KEY` | （无） | Kimi API Key，设置后启动时自动切换至 Kimi 模式 |
| `DEEPSEEK_API_KEY` | （无） | DeepSeek API Key，设置后启动时自动切换至 DeepSeek 模式 |

### 运行时 UI 配置

| 设置项 | 默认值 | 修改位置 |
|--------|--------|---------|
| Ollama API 地址 | `http://192.168.200.54:11434` | LLM 面板 → Ollama → 应用 |
| Ollama 模型 | `qwen3:30b` | LLM 面板 → Ollama → 应用 |
| Kimi 模型 | `moonshot-v1-8k` | LLM 面板 → Kimi → 应用 |
| DeepSeek 模型 | `deepseek-chat` | LLM 面板 → DeepSeek → 应用 |

### 代码配置（`app.py`）

| 设置项 | 代码位置 | 默认值 |
|--------|---------|--------|
| 服务端口 | `ui.run(port=...)` | `8080` |
| 监听地址 | `ui.run(host=...)` | `0.0.0.0` |

---

## 扩展系统

### 添加新产品

1. 在 `agents.py` 中补充 `PRODUCT_PATHS`、`PRODUCT_NAMES_ZH`
2. 在 `TableBuildSkill._DATA` 中为所有 40 个维度添加新产品的值
3. 在 `CITATIONS` 字典中添加新产品各维度的章节+页码引用
4. 更新 `data/products_index.json`
5. 在 `app.py` 中补充 `PRODUCTS`、`PROD_COLORS`、`PDF_URLS`

### 添加新对比维度

1. 在 `agents.py` 的 `DIMENSIONS` 列表追加一个元组：
   ```python
   ("field_key", "维度标签", "B. 疾病覆盖", True, "来源章节兜底提示"),
   ```
2. 在 `TableBuildSkill._DATA["field_key"]` 中添加各产品的值
3. 在 `CITATIONS["field_key"]` 中添加各产品的引用

### 添加新 LLM 后端

1. 在 `mock_llm.py` 的 `LLMBackend` 枚举中添加新值
2. 实现 HTTP 调用函数（参考 `_call_kimi` 的实现模式）
3. 在 `call_llm()` 中添加路由逻辑
4. 在 `app.py` 的 `build_llm_panel()` 中添加 UI 行

---

## 常见问题

**Q：没有 PDF 文件，对比表也能正常生成？**  
A：是的。系统内置了完整的结构化数据层（`TableBuildSkill._DATA` + `CITATIONS`），覆盖全部 5 款产品 × 40 个维度。PDF 只用于增强问答面板的 RAG 检索上下文。

**Q：第一次点"生成对比分析"比较慢，正常吗？**  
A：正常。如果 `data/` 目录中有 PDF，系统会在首次运行时解析并建立 ChromaDB 索引，属于一次性操作。后续运行会秒级恢复已有索引。

**Q：点击"查看原文 PDF"没有新标签页打开？**  
A：NiceGUI 通过 Vue Router 拦截页面内所有 `<a>` 标签的点击事件。本项目用 `data-newwin` 属性配合全局捕获阶段 `addEventListener` 绕过了这一限制。如果仍无效，请检查浏览器的弹出窗口拦截设置。

**Q：如何使用其他 Ollama 模型？**  
A：在 UI 的 LLM 面板中点击 Ollama 行，修改"模型名称"字段后点击**应用**即可立即生效。

**Q：可以部署到云服务器吗？**  
A：可以。代码默认已设置 `host="0.0.0.0"`，开放 8080 端口即可访问。建议在生产环境中配置 nginx 反向代理和 SSL。将 `MOONSHOT_API_KEY` 作为服务器环境变量设置。

**Q：需要哪个 Python 版本？**  
A：Python 3.10 或以上。代码使用了 3.10 引入的内置泛型类型注解语法（`dict[str, str]`）。推荐使用 3.11 以获得更好的性能。

---

## 开源协议

[MIT License](LICENSE) — 可自由使用、修改和分发。
