from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "officeRel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize FDA inspection/citation XLSX exports into benchmark-ready CSV files."
    )
    parser.add_argument(
        "--inspection-xlsx",
        default="benchmark/data/raw/fda_inspections/inspection dataset.xlsx",
    )
    parser.add_argument(
        "--citation-xlsx",
        default="benchmark/data/raw/fda_inspections/citation.xlsx",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/data/raw/fda_inspections/fda_inspections_extract.csv",
    )
    parser.add_argument(
        "--citation-output-csv",
        default="benchmark/data/raw/fda_inspections/fda_citations_extract.csv",
    )
    return parser.parse_args()


def _col_letters_to_index(col: str) -> int:
    result = 0
    for ch in col:
        if ch.isalpha():
            result = result * 26 + (ord(ch.upper()) - 64)
    return result - 1


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", NS)
    if cell_type == "inlineStr":
        text_nodes = cell.findall(".//main:t", NS)
        return "".join(node.text or "" for node in text_nodes)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except Exception:
            return value
    return value


def _read_first_sheet_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", NS):
                text_nodes = item.findall(".//main:t", NS)
                shared_strings.append("".join(node.text or "" for node in text_nodes))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", NS)
        }
        sheets = workbook.find("main:sheets", NS)
        if sheets is None or len(sheets) == 0:
            return []
        first_sheet = sheets[0]
        rel_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_map[rel_id]
        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))

        parsed_rows: list[list[str]] = []
        for row in root.findall(".//main:sheetData/main:row", NS):
            cells: dict[int, str] = {}
            for cell in row.findall("main:c", NS):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                idx = _col_letters_to_index(col)
                cells[idx] = _cell_text(cell, shared_strings).strip()
            if not cells:
                continue
            width = max(cells) + 1
            parsed = [""] * width
            for idx, text in cells.items():
                parsed[idx] = text
            parsed_rows.append(parsed)

        if not parsed_rows:
            return []
        header = parsed_rows[0]
        output: list[dict[str, str]] = []
        for values in parsed_rows[1:]:
            row: dict[str, str] = {}
            for idx, name in enumerate(header):
                if not name:
                    continue
                row[name] = values[idx] if idx < len(values) else ""
            if any(str(value).strip() for value in row.values()):
                output.append(row)
        return output


def _normalize_date_value(value: str) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return ""
    if re.fullmatch(r"\d+(\.\d+)?", text):
        try:
            serial = float(text)
        except ValueError:
            return text
        # Excel serial date base adjusted for the 1900 leap-year bug.
        from datetime import datetime, timedelta

        base = datetime(1899, 12, 30)
        return (base + timedelta(days=serial)).strftime("%Y-%m-%d")
    return text


def _pick_program_area(citation_rows: list[dict[str, str]]) -> str:
    values = [row.get("Program Area", "").strip() for row in citation_rows if row.get("Program Area", "").strip()]
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def main() -> int:
    args = parse_args()
    inspection_path = Path(args.inspection_xlsx)
    citation_path = Path(args.citation_xlsx)
    output_csv = Path(args.output_csv)
    citation_output_csv = Path(args.citation_output_csv)

    inspection_rows = _read_first_sheet_rows(inspection_path)
    citation_rows = _read_first_sheet_rows(citation_path)

    citations_by_inspection: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in citation_rows:
        inspection_id = str(row.get("Inspection ID", "")).strip()
        if inspection_id:
            citations_by_inspection[inspection_id].append(row)

    normalized_rows: list[dict[str, str]] = []
    for row in inspection_rows:
        inspection_id = str(row.get("Inspection ID", "")).strip()
        related_citations = citations_by_inspection.get(inspection_id, [])
        normalized_rows.append(
            {
                "Inspection ID": inspection_id,
                "Classification": str(row.get("Classification", "")).strip(),
                "Program Area": _pick_program_area(related_citations),
                "Project Area": str(row.get("Project Area", "")).strip(),
                "State": str(row.get("State", "")).strip(),
                "Country": str(row.get("Country/Area", "")).strip(),
                "District": "",
                "Fiscal Year": str(row.get("Fiscal Year", "")).strip(),
                "Inspection End Date": _normalize_date_value(row.get("Inspection End Date", "")),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Inspection ID",
                "Classification",
                "Program Area",
                "Project Area",
                "State",
                "Country",
                "District",
                "Fiscal Year",
                "Inspection End Date",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized_rows)

    with citation_output_csv.open("w", encoding="utf-8", newline="") as handle:
        if citation_rows:
            fieldnames = list(citation_rows[0].keys())
        else:
            fieldnames = []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(citation_rows)

    print(
        {
            "inspection_input": str(inspection_path),
            "citation_input": str(citation_path),
            "inspection_rows": len(inspection_rows),
            "citation_rows": len(citation_rows),
            "output_csv": str(output_csv),
            "citation_output_csv": str(citation_output_csv),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
