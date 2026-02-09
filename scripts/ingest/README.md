# Data Ingestion (RAG Setup)

High-quality sources → clean HTML / PDF text → structured chunks → embeddings → store in `chunks` (pgvector).

**Rule:** Fewer documents, higher quality > large noisy corpus.

## What it does

1. **Collect**  
   - **Optional fetch:** Run `python fetch.py` to auto-download from a config (see [Automatic fetch](#automatic-fetch)).  
   - **Manual:** Put `.html` or `.txt` files in **sources/** (PartSelect pages, brand guides; PDF-extracted manuals go in as `.txt`).

2. **Clean**  
   Strips nav/footer/ads, finds main content (`article`, `main`, `.content`), and converts HTML to structured text.

3. **Chunk**  
   Splits by headings into sections and labels each as:
   - **step** – installation/how-to (e.g. “Step 1”, “How to replace…”)
   - **qa** – FAQ / Q&A
   - **symptom** – troubleshooting (e.g. “Not cooling”, “Leaking”)
   - **general** – everything else  

   Long sections are split by paragraph/sentence (max ~3000 chars per chunk). Each chunk has metadata: `section_type`, `title`, `source`, `url`.

4. **Embed**  
   Uses OpenAI `text-embedding-3-small` (1536 dimensions) in batches.

5. **Store**  
   Inserts into `documents` (one per HTML file) and `chunks` (text, metadata, embedding) for vector search.

## Setup

### 1. Database

PostgreSQL with pgvector. From repo root:

```bash
createdb partselect
psql partselect -f apps/api/schema.sql
```

### 2. Python env (ingest only)

```bash
cd scripts/ingest
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Environment

Copy and edit env (required: `DATABASE_URL`, `OPENAI_API_KEY`):

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL and OPENAI_API_KEY
```

Optional:

- `SOURCES_DIR` – folder with `.html` files (default: `scripts/ingest/sources`).

### 4. Add sources

**Option A – Automatic fetch (recommended for many URLs/PDFs)**  
Copy the example config, edit URLs, then run fetch:

```bash
cp sources_config.example.yaml sources_config.yaml
# Edit sources_config.yaml: add html_urls (型号/爆炸图/用户经验页), pdf_urls (安装与使用手册、制造商手册)
python fetch.py
```

This downloads HTML → `sources/*.html`, and PDFs → extracts text → `sources/*.txt`. Then run `python run.py` to chunk and embed.

**Option B – Manual**  
Put high-quality `.html` or `.txt` in `sources/` (or `SOURCES_DIR`):

- PartSelect product/repair pages, brand troubleshooting.
- PDF manuals: extract text to `.txt` (or use fetch.py with `pdf_urls`).

A sample `sources/sample-refrigerator-guide.html` is included so you can run the pipeline once without real pages.

## Automatic fetch

`fetch.py` uses **sources_config.yaml** (see `sources_config.example.yaml`):

- **html_urls:** list of `{ url, name }` – 官网型号页、爆炸图/零件图页、用户经验页等。
- **pdf_urls:** list of `{ url, name }` – 官网或制造商的 PDF 安装/使用手册（含 Troubleshooting 段落）。
- **crawl:** optional `base_url` + `allow_path_prefixes` + `max_pages` – 只抓取指定路径下的页面。
- **fetch_delay_seconds:** 请求间隔，避免对目标站压力过大。

Run `python fetch.py` from `scripts/ingest` (after `pip install -r requirements.txt`). Output is written to `sources/*.html` and `sources/*.txt`; then run `python run.py` to chunk and embed into the vector DB.

**自动化进 sources：** 若目标站对脚本返回 403，fetch 会自动用 **Playwright 无头浏览器**再请求一次，把页面 HTML 拉下来写入 `sources/`，无需手动静态保存。首次使用需安装浏览器内核：`playwright install chromium`。

## Run

From **repo root**:

```bash
python -m scripts.ingest.run
```

Or from `scripts/ingest` (with venv active):

```bash
python run.py
```

Output: number of documents and chunks written. Chunks are then available for RAG retrieval (e.g. vector search in the API).

## Pipeline layout

```
[Optional] fetch.py + sources_config.yaml
    → sources/*.html (scraped) + sources/*.txt (PDF text)

sources/*.html  → html_cleaner → chunker (steps/Q&A/symptoms)
sources/*.txt   → plain_text_to_chunks (section-aware)
    → embedder (OpenAI text-embedding-3-small)
    → db (documents + chunks with embeddings)
```

## 型号数据 (Model number data)

The agent can use **model numbers** to tailor answers and (when wired) to look up compatible parts. The `models` table holds reference data: `model_number`, `brand`, `appliance_type`.

**Option A – Fetch from PartSelect (refrigerator):**

```bash
# From repo root. If the site returns 403, install Playwright browsers once:
playwright install chromium

python -m scripts.ingest.fetch_partselect_models
# Writes config/partselect_refrigerator_models.csv (Parts + Models pages for Whirlpool, GE, Samsung, Frigidaire).
```

**Option A2 – Fetch PartSelect dishwasher model list:**

The list page (https://www.partselect.com/Dishwasher-Models.htm) has ~44k models; the script fetches the first page (100 models) by default. If the site returns 403, use either Playwright or save the page manually:

```bash
# Live fetch (needs playwright install chromium if 403):
python -m scripts.ingest.fetch_partselect_models --appliance dishwasher -o config/partselect_dishwasher_models.csv
```

Or add the URL to `sources_config.yaml` (see example), run `python fetch.py` to save `sources/partselect-dishwasher-models.html`, then:

```bash
python -m scripts.ingest.fetch_partselect_models --from-html sources/partselect-dishwasher-models.html --appliance dishwasher -o config/partselect_dishwasher_models.csv
```

Then seed the DB:

```bash
python -m scripts.ingest.seed_models config/partselect_refrigerator_models.csv
# Or dishwasher:
python -m scripts.ingest.seed_models config/partselect_dishwasher_models.csv
```

**每个型号的零件 (Parts per model):**  
`search_parts(model_number=...)` 依赖 `part_fitment`（型号 ↔ 零件号）和 `parts` 表。直接抓站常因页面 JS 渲染得到 0 条，可用下面两种方式之一：

**方式 1：浏览器保存 Parts 页再解析 (--from-html)**  
打开某型号的零件页，例如 `https://www.partselect.com/Models/3000W10/Parts/`，浏览器「另存为」→ 网页全部/仅 HTML，保存为 `3000W10.html`（文件名即型号）。多个型号可放到同一目录：

```bash
# 目录：每个 型号.html 对应一个型号
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/

# 单文件
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/3000W10.html

# 先试跑不写 DB
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/ --dry-run
```

**方式 2：用 CSV 导入 (--from-csv)**  
若有 (model_number, partselect_number) 数据（手工整理或其它工具导出），可直接导入：

```bash
# CSV 表头: model_number, partselect_number
python -m scripts.ingest.fetch_partselect_model_parts --from-csv config/part_fitment.example.csv
```

示例见 `config/part_fitment.example.csv`。导入后同样会写入 `part_fitment` 和最小 `parts` 行。

**方式 3：Jina / Firecrawl / Bright Data（URL → Markdown → 解析零件号 → 入 DB）**  
用爬虫/反封锁服务请求 PartSelect 零件页，返回 Markdown 后正则提取零件号再写入 DB。

- **--via-brightdata**（推荐）：使用 [Bright Data Web Unlocker](https://github.com/brightdata/brightdata-mcp) 同源 API，专门绕过 403/反爬，免费档约 5000 次/月。需设置 `BRIGHTDATA_API_KEY`，可选 `BRIGHTDATA_ZONE`（默认 `web_unlocker1`）。
- **--via-jina** / **--via-firecrawl**：Jina Reader / Firecrawl；PartSelect 常对二者返回 403，若得到 0 条可改用 --via-brightdata 或方式 1/2。

```bash
# Bright Data（需先在 https://brightdata.com/cp/setting/users 创建 API Key，并创建 Web Unlocker zone 如 web_unlocker1）
export BRIGHTDATA_API_KEY=your-api-key
python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models.csv --limit 5 --dry-run --verbose
```

**方式 4：XML Sitemap（不从首页爬，用 sitemap 拿产品页直达链接）**  
Sitemap 里常有全部产品/型号页的 `<loc>`，可直接解析出型号或 (型号, 零件) 再入 DB。PartSelect 的 sitemap.xml 若被 403，可在浏览器打开 `https://www.partselect.com/sitemap.xml` 另存为 XML 后传本地路径。

```bash
# 传 URL（若 403 会提示保存为文件）
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap https://www.partselect.com/sitemap.xml --output-models-csv config/models_from_sitemap.csv

# 若 sitemap 是 index，可拉取子 sitemap 再汇总
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap https://www.partselect.com/sitemap.xml --sitemap-follow-index 20 --output-models-csv config/models_from_sitemap.csv

# 本地已保存的 sitemap 文件
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap sources/sitemap.xml --output-models-csv config/models_from_sitemap.csv
```

**全量洗碗机：所有型号 + 所有零件**  
从 [Dishwasher-Models.htm](https://www.partselect.com/Dishwasher-Models.htm) 翻页抓全量型号，再对每个型号翻页抓全量零件并写入 DB。需 Bright Data。

```bash
# 一步完成
bash scripts/ingest/fetch_dishwasher_full.sh
# 或分两步：
# 1) python -m scripts.ingest.fetch_partselect_models --appliance dishwasher --via-brightdata --models-max-pages 500 -o config/partselect_dishwasher_models_full.csv
# 2) python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models_full.csv
```

```bash
# Jina Reader（免 key 约 20 RPM；设 JINA_API_KEY 可提高限额）
python -m scripts.ingest.fetch_partselect_model_parts --via-jina config/partselect_dishwasher_models.csv [--limit 5] [--dry-run]

# Firecrawl（需 FIRECRAWL_API_KEY）
python -m scripts.ingest.fetch_partselect_model_parts --via-firecrawl config/partselect_dishwasher_models.csv
```

**Option B – Manual or existing CSV:**

1. Use `config/models_seed.example.csv` or `config/partselect_refrigerator_models.csv` (sample from PartSelect Whirlpool page) as a template.
2. Run the seed script (from repo root, with `DATABASE_URL` set):

   ```bash
   python -m scripts.ingest.seed_models
   # Or: python -m scripts.ingest.seed_models /path/to/your_models.csv
   ```

3. After seeding:
   - **Part lookup:** When you implement `search_parts(model_number=...)` against the DB, use `models` + `compatibility` to return parts that fit the user’s model.
   - **Optional validation:** In `model_parser.py`, you can check `extract_model_number()` results against `models.model_number` to normalize or reject unknown models.

**Do you need model-specific RAG content?**  
Current chunks are **symptom-based** (e.g. “refrigerator too warm”); they are not tagged by model. If you later add **model-specific** repair guides (e.g. “Samsung RF28 not cooling”), you can add a `model_number` or `model_family` field to chunk metadata and filter/boost by it in retrieval when the user provides a model.

## Quality tips

- Prefer **curated** PartSelect and brand guides over dumping whole sites.
- Keep **one logical guide or product page per file** (e.g. one HTML per repair guide or part page).
- After ingestion, run retrieval/eval to tune chunk size and section types if needed.
