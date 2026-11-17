"""Fetch gated Maharashtra datasets (Konkani-Marathi & Vaani-transcription-part) onto S:.

Uses PyArrow HTTP range reads via HfFileSystem (one row-group at a time) to extract ONLY
text and district/language metadata from `ARTPARK-IISc/Vaani-transcription-part` without
downloading audio bytes.

Before running:
1. Accept repository terms on Hugging Face:
   - https://huggingface.co/datasets/sea-rod/konkani-marathi-sentiment-dataset
   - https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part
2. Export a Read token in PowerShell:
   $env:HF_TOKEN = "hf_YOUR_TOKEN"
   python -u "s:\MLAD PROJECT SEM 1\scripts\fetch_gated_konkani_and_vaani.py" --vaani-only
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "datasets")

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from sklearn.model_selection import StratifiedShuffleSplit

RAW_DIR = PROJECT_ROOT / "dataset" / "maharashtra_raw"
OUT_DIR = PROJECT_ROOT / "dataset" / "tier1_backbone"
ANNOT_DIR = PROJECT_ROOT / "dataset" / "annotation_workspace"

VAANI_LANGUAGE_BUCKETS = {
    "Malvani": ("konkan", "malvani"),
    "Konkani": ("konkan", "konkani"),
    "Bhili": ("khandesh", "bhili"),
    "Powari": ("khandesh", "powari"),
    "Gondi": ("vidarbha", "gondi"),
    "Lambani": ("marathwada", "banjari_lambani"),
}

MARATHI_DISTRICT_MAP = {
    "dhule": ("khandesh", "ahirani_khandeshi"),
    "jalgaon": ("khandesh", "ahirani_khandeshi"),
    "nandurbar": ("khandesh", "ahirani_khandeshi"),
    "nagpur": ("vidarbha", "varhadi"),
    "amravati": ("vidarbha", "varhadi"),
    "akola": ("vidarbha", "varhadi"),
    "chandrapur": ("vidarbha", "varhadi"),
    "gondia": ("vidarbha", "varhadi"),
    "washim": ("vidarbha", "varhadi"),
    "yavatmal": ("vidarbha", "varhadi"),
    "sindhudurg": ("konkan", "malvani"),
    "ratnagiri": ("konkan", "konkani_malvani"),
    "aurangabad": ("marathwada", "marathwada_marathi"),
    "nanded": ("marathwada", "marathwada_marathi"),
    "latur": ("marathwada", "marathwada_marathi"),
    "pune": ("western_mh", "deshi_puneri"),
    "satara": ("western_mh", "deshi_satari"),
    "solapur": ("western_mh", "deshi_solapuri"),
    "kolhapur": ("western_mh", "deshi_kolhapuri"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"https?://\S+|www\.\S+", "[URL]", t)
    return " ".join(t.split())


def ensure_hf_auth() -> str | bool:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        who = api.whoami()
        log(f"Authenticated on Hugging Face as: {who.get('name', who.get('fullname', 'user'))}")
        return token or True
    except Exception as exc:
        raise SystemExit(
            "Hugging Face token not found or invalid.\n"
            "Accept terms on:\n"
            "  - https://huggingface.co/datasets/sea-rod/konkani-marathi-sentiment-dataset\n"
            "  - https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part\n"
            "Then run: $env:HF_TOKEN = 'hf_YOUR_TOKEN'"
        ) from exc


def fetch_konkani_marathi(token) -> None:
    """Export Konkani splits + parallel Marathi evaluation file from Konkani-Marathi TSV."""
    local_tsv = RAW_DIR / "konkani_marathi" / "konkani-marathi-sentiment.tsv"
    if not local_tsv.exists():
        local_tsv = Path(
            hf_hub_download(
                "sea-rod/konkani-marathi-sentiment-dataset",
                "konkani-marathi-sentiment.tsv",
                repo_type="dataset",
                local_dir=str(RAW_DIR / "konkani_marathi"),
                token=token,
            )
        )

    df = pd.read_csv(local_tsv, sep="\t")
    cols_lower = {c.lower(): c for c in df.columns}
    kok_col = next(c for k, c in cols_lower.items() if "konkani" in k)
    mr_col = next(c for k, c in cols_lower.items() if "marathi" in k)
    lbl_col = next(c for k, c in cols_lower.items() if "sentiment" in k or "label" in k)

    records, mr_records, seen_kok, seen_mr = [], [], set(), set()
    for _, row in df.iterrows():
        lbl = str(row[lbl_col]).strip().lower()
        if lbl not in {"positive", "negative", "neutral"}:
            continue
        ktxt = norm_text(row[kok_col])
        if ktxt and ktxt not in seen_kok:
            seen_kok.add(ktxt)
            records.append({
                "id": f"konkani_{len(records)}", "text": ktxt, "label": lbl,
                "language": "kok", "dialect": "konkani", "region": "konkan",
                "source": "sea_rod_konkani_marathi",
            })
        mtxt = norm_text(row[mr_col])
        if mtxt and mtxt not in seen_mr:
            seen_mr.add(mtxt)
            mr_records.append({
                "id": f"konkani_mr_{len(mr_records)}", "text": mtxt, "label": lbl,
                "language": "mr", "dialect": "standard_marathi", "region": "konkan",
                "source": "sea_rod_konkani_marathi", "split": "eval_parallel",
            })

    labels = [r["label"] for r in records]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, tmp_idx = next(sss1.split(records, labels))
    tmp_labels = [labels[i] for i in tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    va_pos, te_pos = next(sss2.split(tmp_idx, tmp_labels))

    for sp, idx_list in [
        ("train", tr_idx),
        ("val", [tmp_idx[p] for p in va_pos]),
        ("test", [tmp_idx[p] for p in te_pos]),
    ]:
        out = OUT_DIR / f"konkani_{sp}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for i in idx_list:
                f.write(json.dumps(dict(records[i], split=sp), ensure_ascii=False) + "\n")
        log(f"Exported {len(idx_list)} rows -> {out}")

    mr_out = OUT_DIR / "konkani_parallel_marathi_eval.jsonl"
    with mr_out.open("w", encoding="utf-8") as f:
        for rec in mr_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Exported {len(mr_records)} parallel Marathi rows -> {mr_out}")


def iter_vaani_row_groups(fs: HfFileSystem, repo_path: str) -> Iterator[pd.DataFrame]:
    """Stream non-audio columns one Parquet row group at a time via HTTP range requests."""
    hf_uri = f"datasets/ARTPARK-IISc/Vaani-transcription-part/{repo_path}"
    with fs.open(hf_uri, "rb") as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c.lower() != "audio"]
        log(f"  Opened {repo_path}: row_groups={pf.num_row_groups}, text/meta cols={cols}")
        for rg in range(pf.num_row_groups):
            yield pf.read_row_group(rg, columns=cols).to_pandas()


def pick_text_column(df: pd.DataFrame) -> str | None:
    for cand in ["transcript", "text", "transcription", "sentence", "utterance"]:
        for col in df.columns:
            if cand in col.lower():
                return col
    return None


def shard_priority(path: str) -> tuple[int, str]:
    """Prefer smaller test/validation shards before large train shards."""
    name = path.rsplit("/", 1)[-1]
    if name.startswith("test-"):
        return (0, name)
    if name.startswith("validation-"):
        return (1, name)
    return (2, name)


def fetch_vaani_transcriptions(token, max_per_bucket: int = 350) -> None:
    """Extract Maharashtra dialect & district transcripts cleanly without downloading audio."""
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    files = api.list_repo_files("ARTPARK-IISc/Vaani-transcription-part", repo_type="dataset")
    out_csv = ANNOT_DIR / "vaani_maharashtra_candidates_for_annotation.csv"

    candidates = []
    bucket_counts = collections.Counter()
    seen_texts = set()

    # 1. Dedicated dialect folders (Malvani, Konkani, Bhili, Powari, Gondi, Lambani)
    for lang_folder, (region, dialect) in VAANI_LANGUAGE_BUCKETS.items():
        shards = sorted(
            (f for f in files if f.startswith(f"audio/{lang_folder}/") and f.endswith(".parquet")),
            key=shard_priority,
        )
        for shard in shards:
            if bucket_counts[(region, dialect)] >= max_per_bucket:
                break
            try:
                for df in iter_vaani_row_groups(fs, shard):
                    txt_col = pick_text_column(df)
                    if not txt_col:
                        log(f"  Skipped {shard}: no text col in {list(df.columns)}")
                        break
                    dist_col = next((c for c in df.columns if "district" in c.lower()), None)
                    for _, row in df.iterrows():
                        txt = norm_text(row[txt_col])
                        if len(txt) < 12 or txt in seen_texts:
                            continue
                        seen_texts.add(txt)
                        dist = str(row[dist_col]).strip().lower() if dist_col and pd.notna(row[dist_col]) else lang_folder.lower()
                        candidates.append([
                            f"vaani_{region}_{len(candidates)}", region, dialect, dist,
                            "vaani_transcription", txt, "", "", "", "",
                        ])
                        bucket_counts[(region, dialect)] += 1
                        if bucket_counts[(region, dialect)] >= max_per_bucket:
                            break
                    if bucket_counts[(region, dialect)] >= max_per_bucket:
                        break
            except Exception as exc:
                log(f"  ERROR reading {shard}: {type(exc).__name__}: {exc}")
                if "401" in str(exc) or "403" in str(exc) or "GatedRepo" in type(exc).__name__:
                    raise SystemExit(
                        "Access denied on ARTPARK-IISc/Vaani-transcription-part. "
                        "Click 'Agree and access repository' at "
                        "https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part"
                    ) from exc
        log(f"Folder {lang_folder}: collected {bucket_counts[(region, dialect)]} candidates")

    # 2. Marathi folder filtered by Maharashtra district metadata (test/val shards first)
    mr_shards = sorted(
        (f for f in files if f.startswith("audio/Marathi/") and f.endswith(".parquet")),
        key=shard_priority,
    )
    target_mr_buckets = set(MARATHI_DISTRICT_MAP.values())
    for shard in mr_shards:
        if all(bucket_counts[v] >= max_per_bucket for v in target_mr_buckets):
            break
        try:
            for df in iter_vaani_row_groups(fs, shard):
                txt_col = pick_text_column(df)
                dist_col = next((c for c in df.columns if "district" in c.lower()), None)
                if not txt_col or not dist_col:
                    log(f"  Marathi shard {shard}: missing text/district in {list(df.columns)}")
                    break
                for _, row in df.iterrows():
                    dist_raw = str(row[dist_col]).strip().lower() if pd.notna(row[dist_col]) else ""
                    if dist_raw not in MARATHI_DISTRICT_MAP:
                        continue
                    region, dialect = MARATHI_DISTRICT_MAP[dist_raw]
                    if bucket_counts[(region, dialect)] >= max_per_bucket:
                        continue
                    txt = norm_text(row[txt_col])
                    if len(txt) < 12 or txt in seen_texts:
                        continue
                    seen_texts.add(txt)
                    candidates.append([
                        f"vaani_{region}_{len(candidates)}", region, dialect, dist_raw,
                        "vaani_transcription_mr", txt, "", "", "", "",
                    ])
                    bucket_counts[(region, dialect)] += 1
                if all(bucket_counts[v] >= max_per_bucket for v in target_mr_buckets):
                    break
        except Exception as exc:
            log(f"  ERROR reading {shard}: {type(exc).__name__}: {exc}")

    log(f"Summary by (region, dialect): {dict(bucket_counts)}")
    headers = [
        "id", "region", "dialect", "district", "source_type",
        "text", "label_annotator_1", "label_annotator_2", "final_label", "notes",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(candidates)
    log(f"Saved {len(candidates)} Vaani candidates -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vaani-only", action="store_true", help="Skip Konkani-Marathi export")
    parser.add_argument("--max-per-bucket", type=int, default=350, help="Max candidates per (region, dialect)")
    args = parser.parse_args()

    token = ensure_hf_auth()
    if not args.vaani_only:
        fetch_konkani_marathi(token)
    fetch_vaani_transcriptions(token, max_per_bucket=args.max_per_bucket)


if __name__ == "__main__":
    main()
