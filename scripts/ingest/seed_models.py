#!/usr/bin/env python3
"""
Seed the `models` table with model number reference data.
Use this so the agent can: validate/normalize user-provided model numbers,
and (when wired) look up parts by model via compatibility table.

Usage:
  From repo root:  python -m scripts.ingest.seed_models [path_to_csv]
  Default CSV: config/models_seed.example.csv (or MODELS_SEED_CSV env)

CSV format: model_number,brand,appliance_type
  - model_number: e.g. WRF535SWBM
  - brand: e.g. Whirlpool, GE, Samsung
  - appliance_type: refrigerator | dishwasher
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

scripts_ingest = Path(__file__).resolve().parent
repo_root = scripts_ingest.parents[1]
sys.path.insert(0, str(scripts_ingest))
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(scripts_ingest / ".env")
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

from db import db_connection, get_connection


def load_models_csv(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (model_number, brand, appliance_type)."""
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mn = (row.get("model_number") or "").strip().upper()
            brand = (row.get("brand") or "").strip()
            at = (row.get("appliance_type") or "").strip().lower()
            if not mn or at not in ("refrigerator", "dishwasher"):
                continue
            rows.append((mn, brand, at))
    return rows


def main() -> None:
    csv_path = os.environ.get("MODELS_SEED_CSV")
    if not csv_path and len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if not csv_path:
        csv_path = str(repo_root / "config" / "models_seed.example.csv")
    path = Path(csv_path)
    if not path.is_file():
        print(f"CSV not found: {path}")
        print("Create config/models_seed.example.csv or set MODELS_SEED_CSV. See script doc.")
        sys.exit(1)

    rows = load_models_csv(path)
    if not rows:
        print("No valid rows (model_number, brand, appliance_type).")
        sys.exit(1)

    try:
        conn = get_connection()
        conn.close()
    except Exception as e:
        print(f"DB connection failed: {e}")
        sys.exit(1)

    with db_connection() as conn:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_models_model_number ON models (model_number)")
        for model_number, brand, appliance_type in rows:
            conn.execute(
                """
                INSERT INTO models (model_number, brand, appliance_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_number) DO UPDATE SET
                  brand = EXCLUDED.brand,
                  appliance_type = EXCLUDED.appliance_type
                """,
                (model_number, brand, appliance_type),
            )
        print(f"Upserted {len(rows)} model rows from {path}")

    print("Done. models table is seeded. Use model_number in search_parts / compatibility when wired.")


if __name__ == "__main__":
    main()
