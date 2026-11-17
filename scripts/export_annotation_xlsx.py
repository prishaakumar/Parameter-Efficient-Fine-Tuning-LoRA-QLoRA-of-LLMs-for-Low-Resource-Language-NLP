"""Export Tier-2 CSVs into formatted .xlsx workbooks with dropdowns and color rules."""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT_DIR = Path(__file__).resolve().parent.parent
TIER2_DIR = ROOT_DIR / "dataset" / "tier2_regional_dialect"


def build_xlsx(csv_path: Path, out_xlsx: Path, label_cols: list[str]) -> None:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sentiment_Labelling"
    ws.freeze_panes = "A2"

    hdr_fill = PatternFill("solid", fgColor="0F172A")
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    ws.append(cols)
    for c_idx in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=c_idx)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for r in rows:
        ws.append([r[c] for c in cols])

    text_col_idx = cols.index("text") + 1 if "text" in cols else 1
    for c_idx, name in enumerate(cols, start=1):
        col_letter = openpyxl.utils.get_column_letter(c_idx)
        if name == "text":
            ws.column_dimensions[col_letter].width = 68
        elif "label" in name:
            ws.column_dimensions[col_letter].width = 20
        elif name in ("id", "source_type", "dialect"):
            ws.column_dimensions[col_letter].width = 20
        else:
            ws.column_dimensions[col_letter].width = 15

    for row_idx in range(2, len(rows) + 2):
        cell = ws.cell(row=row_idx, column=text_col_idx)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.font = Font(size=11)

    dv = DataValidation(type="list", formula1='"positive,negative,neutral"', allow_blank=True)
    ws.add_data_validation(dv)

    green = PatternFill("solid", fgColor="CCFBF1")
    red = PatternFill("solid", fgColor="FFE4E6")
    gray = PatternFill("solid", fgColor="E2E8F0")

    for lc in label_cols:
        if lc not in cols:
            continue
        col_letter = openpyxl.utils.get_column_letter(cols.index(lc) + 1)
        rng = f"{col_letter}2:{col_letter}{len(rows) + 1}"
        dv.add(rng)
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"positive"'], fill=green))
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"negative"'], fill=red))
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"neutral"'], fill=gray))

    wb.save(out_xlsx)
    print(f"Saved {out_xlsx} ({len(rows)} rows)")


def main() -> None:
    build_xlsx(
        TIER2_DIR / "tier2_maharashtra_annotation_master.csv",
        TIER2_DIR / "maharashtra_annotation_master.xlsx",
        ["label_annotator_1", "label_annotator_2", "final_label"],
    )


if __name__ == "__main__":
    main()
