-- PartSelect Agent - Core schema (aligned with deep-research-report: structured + RAG)
-- PostgreSQL + pgvector

-- Parts catalog (PartSelect number = PS... or manufacturer number)
CREATE TABLE IF NOT EXISTS parts (
  part_id                  SERIAL PRIMARY KEY,
  part_number              VARCHAR(50) NOT NULL UNIQUE,
  partselect_number        VARCHAR(50),
  manufacturer_part_number VARCHAR(100),
  name                     VARCHAR(255) NOT NULL,
  brand                    VARCHAR(100),
  category                 VARCHAR(100),
  price                    DECIMAL(10, 2),
  stock                    INTEGER DEFAULT 0,
  difficulty               VARCHAR(20),
  time_estimate             VARCHAR(50),
  url                      TEXT,
  image_url                TEXT,
  created_at               TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_parts_partselect ON parts (partselect_number) WHERE partselect_number IS NOT NULL;

-- Appliance models (model page = source; used for part lookup + fitment)
CREATE TABLE IF NOT EXISTS models (
  model_id       SERIAL PRIMARY KEY,
  model_number   VARCHAR(50) NOT NULL,
  brand          VARCHAR(100),
  appliance_type VARCHAR(50) NOT NULL CHECK (appliance_type IN ('refrigerator', 'dishwasher')),
  model_url      TEXT,
  created_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_model_number ON models (model_number);

-- Part–model compatibility (legacy FK; part_fitment is source-of-truth for fitment)
CREATE TABLE IF NOT EXISTS compatibility (
  model_id  INTEGER NOT NULL REFERENCES models(model_id),
  part_id   INTEGER NOT NULL REFERENCES parts(part_id),
  PRIMARY KEY (model_id, part_id)
);

-- Model sections (schematic/diagram per model; section page = source)
CREATE TABLE IF NOT EXISTS model_sections (
  id                SERIAL PRIMARY KEY,
  model_number      VARCHAR(50) NOT NULL,
  section_slug      VARCHAR(100) NOT NULL,
  section_name      VARCHAR(255),
  section_url       TEXT,
  diagram_image_url TEXT,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (model_number, section_slug)
);

-- Section parts (position on diagram → part)
CREATE TABLE IF NOT EXISTS section_parts (
  id                SERIAL PRIMARY KEY,
  model_number      VARCHAR(50) NOT NULL,
  section_slug      VARCHAR(100) NOT NULL,
  diagram_position  VARCHAR(50),
  partselect_number VARCHAR(50) NOT NULL,
  part_name_snapshot VARCHAR(255),
  created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_section_parts_model ON section_parts (model_number, section_slug);

-- Part fitment (deterministic: part fits model; source = part page / Model Cross Reference)
CREATE TABLE IF NOT EXISTS part_fitment (
  id                SERIAL PRIMARY KEY,
  partselect_number VARCHAR(50) NOT NULL,
  model_number      VARCHAR(50) NOT NULL,
  fit_source        VARCHAR(255),
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (partselect_number, model_number)
);
CREATE INDEX IF NOT EXISTS idx_part_fitment_model ON part_fitment (model_number);

-- Part–symptom mapping (from part page "This part fixes the following symptoms")
CREATE TABLE IF NOT EXISTS part_symptoms (
  id                SERIAL PRIMARY KEY,
  partselect_number VARCHAR(50) NOT NULL,
  appliance_type    VARCHAR(50) NOT NULL,
  symptom           VARCHAR(100) NOT NULL,
  source_url        TEXT,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_part_symptoms_symptom ON part_symptoms (appliance_type, symptom);

-- Model symptom rank ("Fixes Symptom xx% of time" per model + symptom)
CREATE TABLE IF NOT EXISTS model_symptom_rank (
  id                SERIAL PRIMARY KEY,
  model_number      VARCHAR(50) NOT NULL,
  symptom           VARCHAR(100) NOT NULL,
  partselect_number VARCHAR(50) NOT NULL,
  fix_rate_text     VARCHAR(50),
  created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_model_symptom_rank ON model_symptom_rank (model_number, symptom);

-- Guides (structured repair/install steps from /Repair/ and blog)
CREATE TABLE IF NOT EXISTS guides (
  guide_id       SERIAL PRIMARY KEY,
  doc_type       VARCHAR(50) NOT NULL,
  appliance_type VARCHAR(50) NOT NULL,
  symptom        VARCHAR(100),
  steps_json     JSONB,
  safety_notes   TEXT,
  source_url     TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_guides_appliance_symptom ON guides (appliance_type, symptom);

-- Documents for RAG
CREATE TABLE IF NOT EXISTS documents (
  doc_id    SERIAL PRIMARY KEY,
  source    VARCHAR(255),
  url       TEXT,
  raw_text  TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks with embeddings (pgvector). Metadata can include: appliance_type, doc_type, brand, model_number, partselect_number, symptom, section_slug (see source_policy)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id   SERIAL PRIMARY KEY,
  doc_id     INTEGER NOT NULL REFERENCES documents(doc_id),
  text       TEXT NOT NULL,
  metadata   JSONB DEFAULT '{}',
  embedding  vector(1536),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Chat logs (optional, for eval/analytics)
CREATE TABLE IF NOT EXISTS chat_logs (
  log_id      SERIAL PRIMARY KEY,
  session_id  VARCHAR(100),
  role        VARCHAR(20),
  content     TEXT,
  scope_label VARCHAR(20),
  created_at  TIMESTAMPTZ DEFAULT NOW()
);
