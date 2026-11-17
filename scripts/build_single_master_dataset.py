"""Consolidate all Tier-1 and Tier-2 datasets into a single unified master dataset.

Outputs:
  - dataset/master/maharashtra_sentiment_single_master.jsonl
  - dataset/master/maharashtra_sentiment_single_master.csv
  - dataset/master/master_dataset_summary.json
"""

from __future__ import annotations

import collections
import csv
import json
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
TIER1_DIR = DATA_DIR / "tier1_backbone"
TIER2_DIR = DATA_DIR / "tier2_regional_dialect"
MASTER_DIR = DATA_DIR / "master"


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def main() -> None:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Tier-2 fewshot split mapping
    adapt_ids = set()
    test_ids = set()
    adapt_p = TIER2_DIR / "tier2_fewshot_adapt.jsonl"
    test_p = TIER2_DIR / "tier2_fewshot_test.jsonl"
    if adapt_p.exists():
        with adapt_p.open("r", encoding="utf-8") as f:
            adapt_ids = {json.loads(line)["id"] for line in f if line.strip()}
    if test_p.exists():
        with test_p.open("r", encoding="utf-8") as f:
            test_ids = {json.loads(line)["id"] for line in f if line.strip()}

    sources_config = [
        # (dir, filename, tier, split, default_lang, default_region, default_dialect, default_source, annotation_type)
        (TIER1_DIR, "mahasent_train.jsonl", "tier_1_backbone", "train", "mr", "standard_marathi", "standard", "l3cube_mahasent_md", "curated_benchmark"),
        (TIER1_DIR, "mahasent_val.jsonl", "tier_1_backbone", "val", "mr", "standard_marathi", "standard", "l3cube_mahasent_md", "curated_benchmark"),
        (TIER1_DIR, "mahasent_test.jsonl", "tier_1_backbone", "test_standard_marathi", "mr", "standard_marathi", "standard", "l3cube_mahasent_md", "curated_benchmark"),
        (TIER1_DIR, "konkani_train.jsonl", "tier_1_backbone", "train", "gom", "konkan_goa", "konkani", "sea_rod_konkani_marathi", "curated_benchmark"),
        (TIER1_DIR, "konkani_val.jsonl", "tier_1_backbone", "val", "gom", "konkan_goa", "konkani", "sea_rod_konkani_marathi", "curated_benchmark"),
        (TIER1_DIR, "konkani_test.jsonl", "tier_1_backbone", "test_konkani", "gom", "konkan_goa", "konkani", "sea_rod_konkani_marathi", "curated_benchmark"),
        (TIER1_DIR, "indicsentiment_mr_eval.jsonl", "tier_1_backbone", "eval_indicsentiment", "mr", "standard_marathi", "standard", "ai4bharat_indicsentiment", "curated_benchmark"),
        (TIER2_DIR, "tier2_dialect_eval.jsonl", "tier_2_regional_dialect", "test_dialect_regional", "mr_dialect", None, None, None, "human_native_annotated"),
    ]

    master_records = []
    seen_ids = set()
    split_counts = collections.Counter()
    label_counts = collections.Counter()
    region_counts = collections.Counter()
    dialect_counts = collections.Counter()
    tier_counts = collections.Counter()

    for src_dir, fname, tier, split_name, def_lang, def_reg, def_dia, def_src, ann_type in sources_config:
        file_path = src_dir / fname
        if not file_path.exists():
            print(f"Skipping missing file: {fname} (not in {src_dir.name})")
            continue

        file_count = 0
        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)

                # Ensure unique, clean ID
                raw_id = str(row.get("id") or f"{fname.replace('.jsonl', '')}_{line_idx}")
                if raw_id in seen_ids:
                    rec_id = f"{fname.replace('.jsonl', '')}_{raw_id}"
                else:
                    rec_id = raw_id
                seen_ids.add(rec_id)

                txt = norm_text(row.get("text", ""))
                lbl = str(row.get("label", "")).strip().lower()
                lang = str(row.get("language") or def_lang).strip().lower()
                reg = str(row.get("region") or def_reg).strip().lower()
                dia = str(row.get("dialect") or def_dia).strip().lower()
                src = str(row.get("source") or def_src).strip().lower()

                # Subsplit metadata for Tier 2
                subsplit = "none"
                if rec_id in adapt_ids or raw_id in adapt_ids:
                    subsplit = "fewshot_adapt"
                elif rec_id in test_ids or raw_id in test_ids:
                    subsplit = "fewshot_test"

                rec = {
                    "id": rec_id,
                    "text": txt,
                    "label": lbl,
                    "split": split_name,
                    "tier": tier,
                    "language": lang,
                    "region": reg,
                    "dialect": dia,
                    "source": src,
                    "annotation_type": ann_type,
                    "subsplit": subsplit,
                }

                master_records.append(rec)
                file_count += 1
                split_counts[split_name] += 1
                label_counts[lbl] += 1
                region_counts[reg] += 1
                dialect_counts[dia] += 1
                tier_counts[tier] += 1

        print(f"Loaded {file_count:6d} rows from {fname}")

    print(f"\nTotal master records compiled: {len(master_records):,}")

    # 1. Export JSONL
    out_jsonl = MASTER_DIR / "maharashtra_sentiment_single_master.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in master_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved Single Master JSONL -> {out_jsonl}")

    # 2. Export CSV
    out_csv = MASTER_DIR / "maharashtra_sentiment_single_master.csv"
    fieldnames = [
        "id",
        "label",
        "split",
        "tier",
        "language",
        "region",
        "dialect",
        "source",
        "annotation_type",
        "subsplit",
        "text",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_records)
    print(f"Saved Single Master CSV   -> {out_csv}")

    # 3. Export Summary JSON
    summary = {
        "dataset_name": "Maharashtra Regional Language & Dialect Sentiment Master Dataset",
        "total_samples": len(master_records),
        "by_tier": dict(tier_counts),
        "by_split": dict(split_counts),
        "by_label": dict(label_counts),
        "by_region": dict(region_counts),
        "by_dialect": dict(dialect_counts),
        "files_compiled": [fname for _, fname, *_ in sources_config],
    }
    out_summary = MASTER_DIR / "master_dataset_summary.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Dataset Summary     -> {out_summary}")


if __name__ == "__main__":
    main()
