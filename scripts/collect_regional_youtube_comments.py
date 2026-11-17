"""Collect Devanagari regional comments from YouTube videos for Maharashtra dialect sentiment annotation.

Uses the official YouTube Data API v3 (`YOUTUBE_API_KEY` env var) to either:
  1. Auto-search top commented regional videos across all 5 Maharashtra regions (--all-regions), OR
  2. Fetch comments from explicit --videos IDs for a chosen --region and --dialect.

Filters strictly to Devanagari-dominant comments (>= 60% Devanagari letters, >= 8 Devanagari chars)
and outputs CSVs matching the Tier-2 human annotation schema.

Usage:
  $env:YOUTUBE_API_KEY = "AIza..."
  python -u "s:\MLAD PROJECT SEM 1\scripts\collect_regional_youtube_comments.py" --all-regions
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "dataset" / "annotation_workspace"
YT_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

DEFAULT_REGION_QUERIES: Dict[str, tuple[str, str, List[str]]] = {
    "khandesh": (
        "ahirani",
        "dhule_jalgaon",
        ["अहिराणी गाणी", "अहिराणी कॉमेडी", "खान्देशी बातम्या", "Ahirani comedy status"],
    ),
    "vidarbha": (
        "varhadi",
        "nagpur_amravati",
        ["वऱ्हाडी कॉमेडी", "वऱ्हाडी नाटक", "विदर्भ बातम्या", "Varhadi comedy"],
    ),
    "konkan": (
        "malvani",
        "sindhudurg_ratnagiri",
        ["मालवणी नाटक", "मालवणी कॉमेडी", "कोकण बातम्या", "Malvani natak"],
    ),
    "marathwada": (
        "marathwada_marathi",
        "aurangabad_nanded",
        ["मराठवाडा बातम्या", "मराठवाडा कॉमेडी", "लातूर नांदेड बातम्या"],
    ),
    "western_mh": (
        "deshi_kolhapuri_satari",
        "satara_kolhapur",
        ["कोल्हापुरी गावरान कॉमेडी", "सातारा बातम्या", "पुणेरी गावरान"],
    ),
}


def clean_comment(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return " ".join(text.split())


def is_devanagari_dominant(text: str, min_deva_chars: int = 8, min_ratio: float = 0.60) -> bool:
    """Keep comments primarily in Devanagari script and exclude pure English/emoji noise."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_deva_chars:
        return False
    deva = sum(1 for c in letters if 0x0900 <= ord(c) <= 0x097F)
    return (deva >= min_deva_chars) and ((deva / len(letters)) >= min_ratio)


def search_video_ids(api_key: str, query: str, max_videos: int = 5) -> List[str]:
    params = {
        "part": "id",
        "q": query,
        "type": "video",
        "regionCode": "IN",
        "relevanceLanguage": "mr",
        "order": "viewCount",
        "maxResults": max_videos,
        "key": api_key,
    }
    resp = requests.get(YT_SEARCH_URL, params=params, timeout=20)
    resp.raise_for_status()
    return [item["id"]["videoId"] for item in resp.json().get("items", []) if "videoId" in item.get("id", {})]


def fetch_video_comments(api_key: str, video_id: str, max_comments: int = 150) -> List[str]:
    comments: List[str] = []
    page_token = None
    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order": "relevance",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(YT_COMMENTS_URL, params=params, timeout=20)
        if resp.status_code == 403:
            # Comments disabled on this specific video; skip gracefully
            break
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            raw = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            cleaned = clean_comment(raw)
            if is_devanagari_dominant(cleaned):
                comments.append(cleaned)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments


def collect_for_region(
    api_key: str,
    region: str,
    dialect: str,
    district: str,
    video_ids: List[str],
    max_per_video: int,
    target_rows: int,
    seen: set[str],
) -> List[list]:
    rows: List[list] = []
    for vid in video_ids:
        if len(rows) >= target_rows:
            break
        for txt in fetch_video_comments(api_key, vid, max_comments=max_per_video):
            if txt in seen:
                continue
            seen.add(txt)
            rows.append([
                f"yt_{region}_{len(rows)}", region, dialect, district.lower(),
                f"youtube:{vid}", txt, "", "", "", "",
            ])
            if len(rows) >= target_rows:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-regions", action="store_true", help="Auto-search and collect across all 5 Maharashtra regions")
    parser.add_argument("--region", choices=list(DEFAULT_REGION_QUERIES))
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--district", default="unknown")
    parser.add_argument("--videos", nargs="+", default=None, help="Explicit YouTube video IDs")
    parser.add_argument("--max-per-video", type=int, default=150)
    parser.add_argument("--target-per-region", type=int, default=350)
    args = parser.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "YOUTUBE_API_KEY not found. Create a YouTube Data API v3 key in Google Cloud Console "
            "and run: $env:YOUTUBE_API_KEY = 'AIza...'"
        )

    headers = [
        "id", "region", "dialect", "district", "source_type",
        "text", "label_annotator_1", "label_annotator_2", "final_label", "notes",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    if args.all_regions:
        all_rows: List[list] = []
        for region, (dialect, district, queries) in DEFAULT_REGION_QUERIES.items():
            vids: List[str] = []
            for q in queries:
                vids.extend(search_video_ids(api_key, q, max_videos=4))
            unique_vids = list(dict.fromkeys(vids))
            reg_rows = collect_for_region(
                api_key, region, dialect, district, unique_vids,
                args.max_per_video, args.target_per_region, seen,
            )
            print(f"[{region}/{dialect}] videos={len(unique_vids)} -> collected {len(reg_rows)} comments", flush=True)
            all_rows.extend(reg_rows)

        out_csv = OUT_DIR / "youtube_maharashtra_candidates_for_annotation.csv"
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} total regional YouTube comments -> {out_csv}", flush=True)
        return

    if not args.region or not args.videos:
        raise SystemExit("Provide --all-regions OR both --region and --videos.")

    dialect = args.dialect or DEFAULT_REGION_QUERIES[args.region][0]
    rows = collect_for_region(
        api_key, args.region, dialect, args.district, args.videos,
        args.max_per_video, args.target_per_region, seen,
    )
    out_csv = OUT_DIR / f"youtube_{args.region}_{dialect}_for_annotation.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved {len(rows)} Devanagari comments -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
