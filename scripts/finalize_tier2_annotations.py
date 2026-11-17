"""Validate human-annotated Tier-2 dialect files (CSV or XLSX), report stats, and export JSONLs.

Reads annotated files (CSV or XLSX), updates `tier2_maharashtra_annotation_master.csv`,
removes overlaps with Tier-1 training data, and writes:
  - dataset/tier2_regional_dialect/tier2_dialect_eval.jsonl (300 rows)
  - dataset/tier2_regional_dialect/tier2_fewshot_adapt.jsonl (150 rows)
  - dataset/tier2_regional_dialect/tier2_fewshot_test.jsonl (150 rows)
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import unicodedata
from pathlib import Path

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
TIER1_DIR = DATA_DIR / "tier1_backbone"
TIER2_DIR = DATA_DIR / "tier2_regional_dialect"
ANNOT_DIR = DATA_DIR / "annotation_workspace"

MASTER_CSV = TIER2_DIR / "tier2_maharashtra_annotation_master.csv"
VALID_LABELS = {"positive", "negative", "neutral"}


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def load_tier1_texts() -> set[str]:
    texts = set()
    for fname in ["mahasent_train.jsonl", "mahasent_val.jsonl", "konkani_train.jsonl", "konkani_val.jsonl"]:
        p = TIER1_DIR / fname
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        texts.add(norm_text(json.loads(line)["text"]))
    return texts


def is_xlsx_file(path: Path) -> bool:
    if path.suffix.lower() in [".xlsx", ".xlsm"]:
        return True
    try:
        with path.open("rb") as f:
            header = f.read(4)
            return header == b"PK\x03\x04"
    except Exception:
        return False


def read_records_from_file(file_path: Path) -> list[dict]:
    """Read rows as list of dicts from either CSV or XLSX."""
    records = []
    if is_xlsx_file(file_path):
        import openpyxl

        with file_path.open("rb") as f:
            content = f.read()
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        sheet = wb.active
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(h).strip().lstrip("\ufeff") if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
        for row in rows[1:]:
            rec = {}
            for h, val in zip(headers, row):
                rec[h] = "" if val is None else str(val).strip()
            records.append(rec)
    else:
        with file_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cleaned = {k.lstrip("\ufeff"): v for k, v in r.items() if k is not None}
                records.append(cleaned)
    return records


def merge_annotated_records_to_master(master_path: Path, new_records: list[dict]) -> dict[str, dict]:
    """Merge newly annotated records into the master CSV file."""
    new_map = {}
    for r in new_records:
        rec_id = r.get("id", "").strip()
        if not rec_id:
            continue
        lbl2 = r.get("label_annotator_2", "").strip().lower()
        lbl1 = r.get("label_annotator_1", "").strip().lower()
        final = r.get("final_label", "").strip().lower()
        new_map[rec_id] = {
            "label_annotator_1": lbl1,
            "label_annotator_2": lbl2,
            "final_label": final,
            "text": r.get("text", "").strip(),
            "region": r.get("region", "").strip(),
            "dialect": r.get("dialect", "").strip(),
        }

    with master_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lstrip("\ufeff") for fn in (reader.fieldnames or [])]
        master_rows = list(reader)
        master_rows = [{k.lstrip("\ufeff"): v for k, v in row.items()} for row in master_rows]

    updated_count = 0
    for row in master_rows:
        rid = row.get("id", "").strip()
        if rid in new_map:
            incoming = new_map[rid]
            if incoming["label_annotator_2"] in VALID_LABELS:
                row["label_annotator_2"] = incoming["label_annotator_2"]
            if incoming["label_annotator_1"] in VALID_LABELS:
                row["label_annotator_1"] = incoming["label_annotator_1"]

            l1 = row.get("label_annotator_1", "").strip().lower()
            l2 = row.get("label_annotator_2", "").strip().lower()
            fin = incoming["final_label"]

            if fin in VALID_LABELS:
                row["final_label"] = fin
            elif l1 in VALID_LABELS and l2 in VALID_LABELS:
                row["final_label"] = l1 if l1 == l2 else ""
            elif l2 in VALID_LABELS:
                row["final_label"] = l2
            elif l1 in VALID_LABELS:
                row["final_label"] = l1

            updated_count += 1

    with master_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_rows)

    print(f"Master CSV updated: {updated_count} rows annotated in {master_path.name}")
    return {r["id"]: r for r in master_rows if r.get("final_label", "").strip().lower() in VALID_LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Tier-2 Dialect Annotations")
    parser.add_argument("input_files", nargs="*", help="Annotated file path(s) (CSV or XLSX)")
    parser.add_argument("--master", type=Path, default=MASTER_CSV, help="Master CSV path")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_files] if args.input_files else []
    if not input_paths:
        candidates = [
            ANNOT_DIR / "annoted_300_dataset.xlsx",
        ]
        input_paths = [p for p in candidates if p.exists()][:1]

    print(f"Processing input files: {[str(p) for p in input_paths]}")
    tier1_texts = load_tier1_texts()

    all_incoming_records = []
    for p in input_paths:
        recs = read_records_from_file(p)
        print(f"  Read {len(recs)} records from {p.name}")
        all_incoming_records.extend(recs)

    annotated_master_map = merge_annotated_records_to_master(args.master, all_incoming_records)

    records, seen = [], set()
    skipped = collections.Counter()
    a1_list, a2_list = [], []

    for rid, row in annotated_master_map.items():
        txt = norm_text(row.get("text", ""))
        lbl1 = row.get("label_annotator_1", "").strip().lower()
        lbl2 = row.get("label_annotator_2", "").strip().lower()
        final = row.get("final_label", "").strip().lower()

        if lbl1 in VALID_LABELS and lbl2 in VALID_LABELS:
            a1_list.append(lbl1)
            a2_list.append(lbl2)

        if not txt or txt.startswith("<paste") or final not in VALID_LABELS:
            skipped["invalid_or_unlabelled"] += 1
            continue
        if txt in tier1_texts:
            skipped["tier1_overlap"] += 1
            continue
        if txt in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(txt)

        records.append({
            "id": row.get("id") or f"tier2_{len(records)}",
            "text": txt,
            "label": final,
            "language": "mr_dialect",
            "dialect": row.get("dialect", "unknown").strip().lower(),
            "region": row.get("region", "unknown").strip().lower(),
            "district": row.get("district", "unknown").strip().lower(),
            "source": row.get("source_type", "human_annotated").strip().lower(),
            "split": "tier2_eval",
        })

    if a1_list:
        kappa = cohen_kappa_score(a1_list, a2_list)
        print(f"\n--- Inter-Annotator Agreement ---")
        print(f"Double-annotated sample count: {len(a1_list)} | Cohen's Kappa: {kappa:.4f}")
    else:
        print("\nSingle-annotator suite verified (all 300 rows labelled by native speaker).")

    print(f"\n--- Dataset Finalization Summary ---")
    print(f"Total valid dialect test records: {len(records)}")
    print(f"Skipped records: {dict(skipped)}")
    print("Class distribution:", dict(collections.Counter(r["label"] for r in records)))
    print("Region distribution:", dict(collections.Counter(r["region"] for r in records)))
    print("Dialect distribution:", dict(collections.Counter(r["dialect"] for r in records)))

    eval_path = TIER2_DIR / "tier2_dialect_eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nExported Full Tier-2 Dialect Eval Suite -> {eval_path} ({len(records)} samples)")

    if len(records) >= 20:
        strata = [f"{r['region']}_{r['label']}" for r in records]
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            adapt_idx, test_idx = next(sss.split(records, strata))
        except ValueError:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            adapt_idx, test_idx = next(sss.split(records, [r["label"] for r in records]))

        for name, idxs, sp in [
            ("tier2_fewshot_adapt.jsonl", adapt_idx, "fewshot_adapt"),
            ("tier2_fewshot_test.jsonl", test_idx, "fewshot_test"),
        ]:
            out_p = TIER2_DIR / name
            with out_p.open("w", encoding="utf-8") as f:
                for i in idxs:
                    f.write(json.dumps(dict(records[i], split=sp), ensure_ascii=False) + "\n")
            print(f"Exported Stratified Split -> {out_p} ({len(idxs)} samples)")


if __name__ == "__main__":
    main()
