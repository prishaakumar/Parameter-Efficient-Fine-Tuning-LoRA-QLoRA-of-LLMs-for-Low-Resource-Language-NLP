"""Preprocess and export Tier-1 Maharashtra labelled sentiment datasets onto S:.

Outputs leak-free, deduplicated, NFC-normalized JSONL files for:
1. L3Cube-MahaSent-MD (3-class Standard Marathi across 4 domains + nested 10/25/50/100% ablations)
2. AI4Bharat IndicSentiment (mr, 2-class external Positive/Negative generalization set)
3. Tier-2 5-region Maharashtra dialect annotation template CSV
"""

from __future__ import annotations

import collections
import csv
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "maharashtra_raw"
OUT_DIR = PROJECT_ROOT / "dataset" / "tier1_backbone"
ABL_DIR = OUT_DIR / "ablations"

LABEL_MAP_MAHA = {-1: "negative", 0: "neutral", 1: "positive"}
VALID_LABELS = {"positive", "negative", "neutral"}

DOMAINS = [
    ("MahaSent_GT", "tweets-train.csv", "tweets-valid.csv", "tweets-test.csv", "tweet", "general_tweet"),
    ("MahaSent_MR", "MahaSent_MR_Train.csv", "MahaSent_MR_Val.csv", "MahaSent_MR_Test.csv", "marathi_sentence", "movie_review"),
    ("MahaSent_PT", "tweets-train.csv", "tweets-valid.csv", "tweets-test.csv", "tweet", "political_tweet"),
    ("MahaSent_ST", "MahaSent_ST_Train.csv", "MahaSent_ST_Val.csv", "MahaSent_ST_Test.csv", "marathi_text", "subtitle"),
]


def clean_text(text: object) -> str:
    """Apply LLM-safe NFC normalization and normalize URLs/whitespace."""
    if not isinstance(text, str):
        return ""
    normed = unicodedata.normalize("NFC", text)
    normed = re.sub(r"https?://\S+|www\.\S+", "[URL]", normed)
    return " ".join(normed.split())


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write an iterable of dict records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize_records(records: List[dict]) -> dict:
    """Return exact counts by label and domain for verification."""
    return {
        "count": len(records),
        "labels": dict(collections.Counter(r["label"] for r in records)),
        "domains": dict(collections.Counter(r.get("domain", "unknown") for r in records)),
    }


def load_mahasent_raw() -> Tuple[Dict[str, List[Tuple[str, str, str]]], set]:
    """Load all 4 MahaSent-MD domains and detect cross-dataset label conflicts."""
    maha_root = RAW_DIR / "L3Cube-MahaSent-MD"
    raw_splits: Dict[str, List[Tuple[str, str, str]]] = {"train": [], "val": [], "test": []}
    seen_labels: Dict[str, str] = {}
    conflicts: set = set()

    for folder, ftr, fva, fte, text_col, domain in DOMAINS:
        for split_name, fname in [("train", ftr), ("val", fva), ("test", fte)]:
            df = pd.read_csv(maha_root / folder / fname)
            for _, row in df.iterrows():
                txt = clean_text(row[text_col])
                lbl = LABEL_MAP_MAHA.get(row["label"])
                if not txt or lbl not in VALID_LABELS:
                    continue
                if txt in seen_labels and seen_labels[txt] != lbl:
                    conflicts.add(txt)
                else:
                    seen_labels[txt] = lbl
                raw_splits[split_name].append((txt, lbl, domain))

    return raw_splits, conflicts


def deduplicate_splits(
    raw_splits: Dict[str, List[Tuple[str, str, str]]],
    conflicts: set,
) -> Dict[str, List[dict]]:
    """Deduplicate test -> val -> train in strict priority order to eliminate leakage."""
    locked_texts: set = set()
    cleaned: Dict[str, List[dict]] = {}

    for split_name in ["test", "val", "train"]:
        split_recs: List[dict] = []
        split_seen: set = set()
        for txt, lbl, domain in raw_splits[split_name]:
            if txt in conflicts or txt in locked_texts or txt in split_seen:
                continue
            split_seen.add(txt)
            split_recs.append({
                "id": f"mahasent_{split_name}_{len(split_recs)}",
                "text": txt,
                "label": lbl,
                "language": "mr",
                "dialect": "standard_marathi",
                "region": "across_maharashtra",
                "domain": domain,
                "source": "l3cube_mahasent_md",
                "split": split_name,
            })
        locked_texts.update(split_seen)
        cleaned[split_name] = split_recs

    return cleaned


def build_nested_ablations(train_records: List[dict], seed: int = 42) -> Dict[str, List[dict]]:
    """Create nested stratified subsets (10% < 25% < 50% < 100%) balanced by domain+label."""
    strata = [f"{r['domain']}_{r['label']}" for r in train_records]
    indices = list(range(len(train_records)))

    def subsample(pool_idx: List[int], keep_ratio_of_pool: float) -> List[int]:
        pool_strata = [strata[i] for i in pool_idx]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=keep_ratio_of_pool, random_state=seed)
        sub_pos, _ = next(sss.split(pool_idx, pool_strata))
        return sorted(pool_idx[p] for p in sub_pos)

    idx_50 = subsample(indices, 0.50)
    idx_25 = subsample(idx_50, 0.50)   # 50% of 50% = 25% of total
    idx_10 = subsample(idx_25, 0.40)   # 40% of 25% = 10% of total

    return {
        "10pct": [train_records[i] for i in idx_10],
        "25pct": [train_records[i] for i in idx_25],
        "50pct": [train_records[i] for i in idx_50],
        "100pct": train_records,
    }


def prepare_indicsentiment_mr(forbidden_texts: set) -> List[dict]:
    """Load IndicSentiment Marathi validation+test, drop null labels & overlaps."""
    records: List[dict] = []
    seen: set = set()
    for sp in ["validation", "test"]:
        path = RAW_DIR / "IndicSentiment" / "data" / sp / "mr.json"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_lbl = row.get("LABEL")
                if not isinstance(raw_lbl, str):
                    continue
                lbl = raw_lbl.strip().lower()
                txt = clean_text(row.get("INDIC REVIEW", ""))
                if not txt or lbl not in VALID_LABELS or txt in forbidden_texts or txt in seen:
                    continue
                seen.add(txt)
                records.append({
                    "id": f"indicsent_mr_{sp}_{len(records)}",
                    "text": txt,
                    "label": lbl,
                    "language": "mr",
                    "dialect": "standard_marathi",
                    "region": "across_maharashtra",
                    "domain": str(row.get("CATEGORY", "product_review")).lower(),
                    "source": "ai4bharat_indicsentiment",
                    "split": "eval_binary",
                })
    return records


def write_tier2_template(path: Path) -> None:
    """Create a blank human-annotation CSV template for the 5 Maharashtra regions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "id", "region", "dialect", "district", "source_type",
        "text", "label_annotator_1", "label_annotator_2", "final_label", "notes",
    ]
    sample_guide_rows = [
        ["khandesh_0001", "khandesh", "ahirani", "dhule", "vaani_or_youtube", "<paste Ahirani Devanagari sentence>", "", "", "", "positive | negative | neutral"],
        ["vidarbha_0001", "vidarbha", "varhadi", "nagpur", "vaani_or_youtube", "<paste Varhadi Devanagari sentence>", "", "", "", "positive | negative | neutral"],
        ["konkan_0001", "konkan", "malvani", "sindhudurg", "vaani_or_youtube", "<paste Malvani Devanagari sentence>", "", "", "", "positive | negative | neutral"],
        ["marathwada_0001", "marathwada", "marathwada_marathi", "aurangabad", "vaani_or_youtube", "<paste Marathwada sentence>", "", "", "", "positive | negative | neutral"],
        ["western_mh_0001", "western_mh", "deshi_satari", "satara", "vaani_or_youtube", "<paste Western MH sentence>", "", "", "", "positive | negative | neutral"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(sample_guide_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_splits, conflicts = load_mahasent_raw()
    cleaned = deduplicate_splits(raw_splits, conflicts)

    for sp in ["train", "val", "test"]:
        write_jsonl(cleaned[sp], OUT_DIR / f"mahasent_{sp}.jsonl")

    ablations = build_nested_ablations(cleaned["train"], seed=42)
    for tag, recs in ablations.items():
        write_jsonl(recs, ABL_DIR / f"mahasent_train_{tag}.jsonl")

    all_maha_texts = {r["text"] for sp in cleaned.values() for r in sp}
    indic_mr = prepare_indicsentiment_mr(all_maha_texts)
    write_jsonl(indic_mr, OUT_DIR / "indicsentiment_mr_eval.jsonl")

    write_tier2_template(OUT_DIR / "tier2_dialect_annotation_template.csv")

    summary = {
        "mahasent_conflicts_dropped": len(conflicts),
        "mahasent_splits": {sp: summarize_records(cleaned[sp]) for sp in ["train", "val", "test"]},
        "mahasent_ablations": {tag: len(recs) for tag, recs in ablations.items()},
        "indicsentiment_mr_eval": summarize_records(indic_mr),
    }
    with (OUT_DIR / "tier1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

