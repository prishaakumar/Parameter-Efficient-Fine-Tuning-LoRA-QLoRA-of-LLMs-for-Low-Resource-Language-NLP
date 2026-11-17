# Parameter-Efficient Fine-Tuning (LoRA / QLoRA) for Maharashtra Regional Sentiment Analysis

**Academic Project & Research Thesis Repository**  
**Focus:** Standard Marathi, Konkani, and 5 Maharashtra Regional Dialects (*Ahirani, Varhadi, Malvani, Marathwada, and Deshi*)

---

## 1. Repository Directory Structure

```
S:\MLAD PROJECT SEM 1\
│
├── README.md                                  # Comprehensive Project Guide & Repository Map
├── requirements.txt                           # Python Dependencies
├── .gitignore                                 # Git Ignore Rules
│
├── dataset/                                   # All Structured & Validated Datasets
│   ├── master/                                # Unified Consolidated Master Dataset (67,800 Rows)
│   │   ├── maharashtra_sentiment_single_master.csv    # Single CSV (for Excel / Pandas)
│   │   ├── maharashtra_sentiment_single_master.jsonl  # Single JSONL (for PyTorch / HuggingFace)
│   │   └── master_dataset_summary.json                # Statistical Audit & Partition Metadata
│   │
│   ├── tier1_backbone/                        # High-Volume Training Backbone (67,500 Rows)
│   │   ├── mahasent_train.jsonl               # Standard Marathi Train (47,647 rows)
│   │   ├── mahasent_val.jsonl                 # Standard Marathi Validation (5,968 rows)
│   │   ├── mahasent_test.jsonl                # Standard Marathi Test Holdout (6,734 rows)
│   │   ├── konkani_train.jsonl                # Konkani Belt Train (4,199 rows)
│   │   ├── konkani_val.jsonl                  # Konkani Belt Validation (900 rows)
│   │   ├── konkani_test.jsonl                 # Konkani Belt Test Holdout (900 rows)
│   │   ├── indicsentiment_mr_eval.jsonl       # External Binary Evaluation (1,152 rows)
│   │   └── ablations/                         # Data-Efficiency Subsets (10%, 25%, 50%, 100%)
│   │
│   ├── tier2_regional_dialect/                # Authentic Human-Annotated Benchmark (300 Rows)
│   │   ├── tier2_dialect_eval.jsonl           # Primary Test Suite (60 rows per region)
│   │   ├── tier2_fewshot_adapt.jsonl          # 50% Few-Shot In-Context Split (150 rows)
│   │   ├── tier2_fewshot_test.jsonl           # 50% Few-Shot Holdout Test (150 rows)
│   │   ├── maharashtra_annotation_master.xlsx # Formatted Excel sheet with color formatting
│   │   └── tier2_maharashtra_annotation_master.csv  # 2,480 Candidate Pool with annotations
│   │
│   ├── annotation_workspace/                  # Candidate Pools & Completed Annotation Sheets
│   │   ├── annoted_300_dataset.xlsx           # Completed human annotation sheet (300 rows)
│   │   ├── youtube_maharashtra_candidates_for_annotation.csv  # 1,750 Scraped Comments
│   │   └── vaani_maharashtra_candidates_for_annotation.csv    # 730 IISc Vaani Transcriptions
│   │
│   └── maharashtra_processed/                 # Backward-Compatible Processed Data Directory
│
├── docs/                                      # Official PDF Reports for Faculty Guide & Thesis
│   ├── Master_Dataset_Architecture_and_Preprocessing_Report.pdf  # 5-Page Master Technical Report
│   ├── Model_Training_and_Research_Implementation_Plan.pdf       # 3-Page Training & Paper Roadmap
│   ├── Maharashtra_Project_Summary_and_Data_Collection_Report.pdf # 4-Page Initial Collection Report
│   ├── ML_Dataset_Audit_and_Preprocessing_Guide.pdf             # Comprehensive Preprocessing Guide
│   └── html_sources/                          # HTML Source files for document compilation
│
├── scripts/                                   # Automated Pipeline Python Scripts
│   ├── collect_regional_youtube_comments.py   # Regional YouTube API scraper
│   ├── fetch_gated_konkani_and_vaani.py       # IISc Vaani parquet streaming
│   ├── prepare_maharashtra_tier1.py           # MahaSent & IndicSentiment preprocessor
│   ├── finalize_tier2_annotations.py          # Annotation validation & JSONL exporter
│   ├── build_single_master_dataset.py         # 67,800-row master dataset compiler
│   ├── export_annotation_xlsx.py              # Excel workbook formatting & validation
│   ├── generate_master_data_report_pdf.py     # Master dataset report generator
│   └── generate_implementation_plan_pdf.py    # Implementation plan PDF generator
│
└── results/                                   # Model Checkpoints, Logs, and Evaluation Outputs
```

---

## 2. Dataset Overview (67,800 Total Observations)

| Split Tag (`split`) | Observations | Language Variety | Tier | Research Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`train`** | **51,846** | Standard Marathi (47.6k) + Konkani (4.2k) | Tier-1 | Backbone LoRA & QLoRA training |
| **`val`** | **6,868** | Standard Marathi (5.9k) + Konkani (0.9k) | Tier-1 | Validation loss & early stopping |
| **`test_standard_marathi`** | **6,734** | Standard Marathi (MahaSent-MD) | Tier-1 | In-domain Standard Marathi test holdout |
| **`test_konkani`** | **900** | Konkani Belt (Goa & Sindhudurg) | Tier-1 | Coastal Konkani generalization test |
| **`eval_indicsentiment`** | **1,152** | AI4Bharat IndicSentiment (Marathi) | Tier-1 | External cross-distribution evaluation |
| **`test_dialect_regional`** | **300** | **5 Maharashtra Regional Dialects** | **Tier-2** | **Core Human Dialect Test Benchmark** |
| **Total Master Dataset** | **67,800** | Balanced: 23.3k Pos / 20.8k Neu / 23.7k Neg | — | **Single Consolidated Master** |

---

## 3. Tier-2 Regional Benchmark (300 Human-Verified Rows)

Every cultural region of Maharashtra has **exactly 60 verified test samples**:
* **Khandesh (North MH):** 27 Positive, 33 Neutral, 0 Negative (*Ahirani, Bhili*)
* **Vidarbha (East MH):** 9 Positive, 48 Neutral, 3 Negative (*Varhadi, Powari*)
* **Konkan (Coastal MH):** 12 Positive, 44 Neutral, 4 Negative (*Malvani*)
* **Marathwada (Central MH):** 16 Positive, 36 Neutral, 8 Negative (*Marathwada Marathi*)
* **Western Maharashtra:** 13 Positive, 47 Neutral, 0 Negative (*Deshi, Puneri, Kolhapuri*)

---

## 4. Academic Quality & Research Validation
* **Zero Train-Test Contamination:** 0.000% string overlap between training and all test splits.
* **Unicode NFC Normalized:** 100% Devanagari ligatures and glyph encodings unified.
* **LLM Syntax Preserved:** Emojis and grammatical negation particles (*नाही, नको*) retained for optimal transformer attention.
* **Context Length:** Sequence length `max_seq_len = 256` prevents out-of-memory errors with 0% sentence truncation.

---

## 5. Model Training & Dataset Loading

The datasets can be loaded directly using Hugging Face `datasets` or `pandas`:

```python
from datasets import load_dataset
import pandas as pd

# 1. Load the Single Consolidated Master Dataset
dataset = load_dataset("json", data_files="dataset/master/maharashtra_sentiment_single_master.jsonl")

# 2. Or load training backbone & dialect benchmark separately
train_df = pd.read_json("dataset/tier1_backbone/mahasent_train.jsonl", lines=True)
test_dialect_df = pd.read_json("dataset/tier2_regional_dialect/tier2_dialect_eval.jsonl", lines=True)

print(f"Loaded {len(train_df)} training samples and {len(test_dialect_df)} dialect test samples.")
```

