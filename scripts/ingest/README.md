# Data ingestion (RAG setup)

Pipeline: **sources** → clean HTML/text → **chunk** (by section type) → **embed** (OpenAI) → **store** in Postgres (pgvector).

**Principle:** Prefer fewer, high-quality documents over a large noisy corpus.

---

## What it does

1. **Collect**
   - **Optional fetch:** Run `python fetch.py` to download content from a YAML config (see [Automatic fetch](#automatic-fetch)).
   - **Manual:** Place `.html` or `.txt` files in **sources/** (e.g. PartSelect repair pages, brand guides; PDF-extracted text as `.txt`).

2. **Clean**
   Strips nav/footer/ads, finds main content (`article`, `main`, `.content`), and converts HTML to structured text.

3. **Chunk**
   Splits by headings and labels each section as:
   - **step** – installation / how-to (e.g. “Step 1”, “How to replace…”)
   - **qa** – FAQ / Q&A
   - **symptom** – troubleshooting (e.g. “Not cooling”, “Leaking”)
   - **general** – everything else

   Long sections are split by paragraph/sentence (max ~3000 chars per chunk). Each chunk has metadata: `section_type`, `title`, `source`, `url`.

4. **Embed**
   Uses OpenAI `text-embedding-3-small` (1536 dimensions) in batches.

5. **Store**
   Inserts into `documents` (one per source file) and `chunks` (text, metadata, embedding) for vector search in the API.

---

## Setup

### 1. Database

PostgreSQL with pgvector. From repo root:

```bash
createdb partselect
psql partselect -f apps/api/schema.sql
```

### 2. Python environment (ingest only)

```bash
cd scripts/ingest
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment

Copy the example env and set required variables:

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL and OPENAI_API_KEY
```

Optional:

- **SOURCES_DIR** – directory containing `.html`/`.txt` files (default: `scripts/ingest/sources`).

### 4. Add sources

**Option A – Automatic fetch (recommended for many URLs)**  
Copy the example config, add URLs, then run fetch:

```bash
cp sources_config.example.yaml sources_config.yaml
# Edit sources_config.yaml: add html_urls (repair pages, model lists), pdf_urls (manuals)
python fetch.py
```

This downloads HTML → `sources/*.html`, and PDFs → extracted text → `sources/*.txt`. Then run `python run.py` to chunk and embed.

**Option B – Manual**  
Place `.html` or `.txt` files in `sources/` (or `SOURCES_DIR`):

- PartSelect repair/product pages, brand troubleshooting guides.
- For PDF manuals: extract text to `.txt`, or use `fetch.py` with `pdf_urls` in the config.

Sample files in `sources/` are included so you can run the pipeline once without adding your own pages.

---

## Automatic fetch

`fetch.py` reads **sources_config.yaml** (see `sources_config.example.yaml`):

- **html_urls:** list of `{ url, name }` – repair pages, model list pages, symptom pages, etc.
- **pdf_urls:** list of `{ url, name }` – PDF manuals (e.g. installation, troubleshooting).
- **crawl:** optional `base_url` + `allow_path_prefixes` + `max_pages` – limit crawling to certain path prefixes.
- **fetch_delay_seconds:** delay between requests to avoid overloading the target site.

Run `python fetch.py` from `scripts/ingest` (after `pip install -r requirements.txt`). Output goes to `sources/*.html` and `sources/*.txt`; then run `python run.py` to chunk and embed into the database.

**403 / blocked responses:** If the target site returns 403 to plain HTTP, fetch can retry with **Playwright** (headless Chromium) and write the HTML to `sources/`. Install the browser once: `playwright install chromium`.

---

## Run the pipeline

From **repo root**:

```bash
python -m scripts.ingest.run
```

Or from `scripts/ingest` (with venv active):

```bash
python run.py
```

Output: number of documents and chunks written. Chunks are then used by the API for RAG retrieval (vector search).

---

## Pipeline layout

```
[Optional] fetch.py + sources_config.yaml
    → sources/*.html (scraped) + sources/*.txt (PDF text)

sources/*.html  → html_cleaner → chunker (step / qa / symptom / general)
sources/*.txt   → plain_text_to_chunks (section-aware)
    → embedder (OpenAI text-embedding-3-small)
    → db (documents + chunks with embeddings)
```

---

## Model number data

The agent uses **model numbers** to tailor answers and (when the DB is populated) to look up compatible parts. The `models` table stores: `model_number`, `brand`, `appliance_type`.

### Option A – Fetch from PartSelect (refrigerator)

```bash
# From repo root. If the site returns 403, install Playwright once:
playwright install chromium

python -m scripts.ingest.fetch_partselect_models
# Writes config/partselect_refrigerator_models.csv (Whirlpool, GE, Samsung, Frigidaire, etc.).
```

### Option A2 – Fetch PartSelect dishwasher model list

The list page (https://www.partselect.com/Dishwasher-Models.htm) has many models; the script fetches the first page (100 models) by default. If the site returns 403, use Playwright or save the page manually:

```bash
# Live fetch (install Playwright if you get 403):
python -m scripts.ingest.fetch_partselect_models --appliance dishwasher -o config/partselect_dishwasher_models.csv
```

Or add the list URL to `sources_config.yaml`, run `python fetch.py` to save e.g. `sources/partselect-dishwasher-models.html`, then:

```bash
python -m scripts.ingest.fetch_partselect_models --from-html sources/partselect-dishwasher-models.html --appliance dishwasher -o config/partselect_dishwasher_models.csv
```

Then seed the DB:

```bash
python -m scripts.ingest.seed_models config/partselect_refrigerator_models.csv
# Or dishwasher:
python -m scripts.ingest.seed_models config/partselect_dishwasher_models.csv
```

### Parts per model (part_fitment)

`search_parts(model_number=...)` in the API relies on `part_fitment` (model ↔ part number) and the `parts` table. Direct scraping often returns 403 or empty results due to JS-rendered pages. You can use any of the following.

**Method 1 – Save Parts pages in the browser, then parse (--from-html)**  
Open a model’s parts page (e.g. `https://www.partselect.com/Models/3000W10/Parts/`), save as “Webpage, complete” or HTML only, e.g. as `3000W10.html` (filename = model number). Put one or more such files in a directory:

```bash
# Directory: one <model>.html per model
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/

# Single file
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/3000W10.html

# Dry run (no DB writes)
python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/ --dry-run
```

**Method 2 – Import from CSV (--from-csv)**  
If you have (model_number, partselect_number) data from another tool or manual export:

```bash
# CSV header: model_number, partselect_number
python -m scripts.ingest.fetch_partselect_model_parts --from-csv config/part_fitment.example.csv
```

Example: `config/part_fitment.example.csv`. This populates `part_fitment` and minimal `parts` rows.

**Method 3 – Bright Data / Jina / Firecrawl (URL → fetch → parse → DB)**  
Use a scraping or “web unlocker” service to fetch PartSelect parts pages, then parse part numbers and write to the DB.

- **--via-brightdata** (recommended): [Bright Data Web Unlocker](https://github.com/brightdata/brightdata-mcp) – good for bypassing 403; free tier ~5000 requests/month. Set `BRIGHTDATA_API_KEY` in `.env`; optional `BRIGHTDATA_ZONE` (default `web_unlocker1`).
- **--via-jina** / **--via-firecrawl**: Jina Reader or Firecrawl; PartSelect often returns 403 to these; if you get no results, try --via-brightdata or Method 1/2.

```bash
# Bright Data (create API key and Web Unlocker zone at https://brightdata.com)
export BRIGHTDATA_API_KEY=your-api-key
python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models.csv --limit 5 --dry-run --verbose
```

**Method 4 – XML sitemap (--from-sitemap)**  
Sitemaps often list product/model URLs. If the live sitemap returns 403, open `https://www.partselect.com/sitemap.xml` in a browser and save as a local file.

```bash
# From URL (script will prompt to save locally if 403)
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap https://www.partselect.com/sitemap.xml --output-models-csv config/models_from_sitemap.csv

# Follow sitemap index and fetch sub-sitemaps
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap https://www.partselect.com/sitemap.xml --sitemap-follow-index 20 --output-models-csv config/models_from_sitemap.csv

# From a local sitemap file
python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap sources/sitemap.xml --output-models-csv config/models_from_sitemap.csv
```

**Full dishwasher run (all models + all parts)**  
Fetch all models from [Dishwasher-Models.htm](https://www.partselect.com/Dishwasher-Models.htm), then fetch parts for each and write to the DB. Requires Bright Data.

```bash
# One-shot script
bash scripts/ingest/fetch_dishwasher_full.sh
# Or manually:
# 1) python -m scripts.ingest.fetch_partselect_models --appliance dishwasher --via-brightdata --models-max-pages 500 -o config/partselect_dishwasher_models_full.csv
# 2) python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models_full.csv
```

```bash
# Jina Reader (free tier ~20 RPM; set JINA_API_KEY for higher limits)
python -m scripts.ingest.fetch_partselect_model_parts --via-jina config/partselect_dishwasher_models.csv [--limit 5] [--dry-run]

# Firecrawl (requires FIRECRAWL_API_KEY)
python -m scripts.ingest.fetch_partselect_model_parts --via-firecrawl config/partselect_dishwasher_models.csv
```

### Option B – Manual or existing CSV

1. Use `config/models_seed.example.csv` or the generated `config/partselect_refrigerator_models.csv` as a template.
2. With `DATABASE_URL` set, from repo root:

   ```bash
   python -m scripts.ingest.seed_models
   # Or: python -m scripts.ingest.seed_models /path/to/your_models.csv
   ```

3. After seeding, the API can use `models` and compatibility data for part lookup and (optionally) model validation in `model_parser.py`.

**Model-specific RAG:** Current chunks are **symptom-based** (e.g. “refrigerator too warm”), not tagged by model. If you add model-specific repair guides later, you can add `model_number` or `model_family` to chunk metadata and filter/boost by it in retrieval when the user provides a model.

---

## Quality tips

- Prefer **curated** PartSelect and brand guides over dumping entire sites.
- Keep **one logical guide or product page per file** (e.g. one HTML per repair guide or part page).
- After ingestion, you can tune chunk size and section types based on retrieval quality.
