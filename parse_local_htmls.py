"""
Parse student destination data from locally-saved HTML files.

The TJ destinations site requires sign-in, so pages are downloaded manually
into the htmls/ folder and parsed here.

Each page is a single <table class="table"> with one <tr class="d-flex">
per student. Each row has five cells:
    1. Name
    2. GPA
    3. Test scores — nested <table> of (Test, Score) pairs
    4. Biography — <div id="biography"> with free-form HTML
    5. Decisions — nested <table> of (Type+Result icon, College + Location)

The output CSV is one row per student. Complex columns (test_scores,
decisions) are JSON-encoded so they survive the CSV round-trip cleanly.
Names are still real here — anonymization is the next pipeline step.
"""

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

HTML_DIR = Path("htmls")
OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "raw_destinations.csv"


def parse_test_scores(cell: Tag) -> dict[str, str]:
    scores: dict[str, str] = {}
    for tr in cell.select("table tbody tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            test = tds[0].get_text(strip=True)
            score = tds[1].get_text(strip=True)
            if test:
                scores[test] = score
    return scores


def parse_biography(cell: Tag) -> str:
    bio = cell.find("div", id="biography") or cell
    # separator="\n" preserves paragraph + list-item breaks
    return bio.get_text(separator="\n", strip=True)


def parse_decisions(cell: Tag) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for tr in cell.select("table tbody tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 2:
            continue

        type_span = tds[0].find("span")
        decision_type = type_span.get_text(strip=True) if type_span else ""

        result_icon = tds[0].find("i")
        result = (
            result_icon.get("aria-label", "").strip()
            if result_icon is not None
            else ""
        )

        # College cell uses <br> between name and location.
        college_text = tds[1].get_text(separator="\n", strip=True)
        parts = [p.strip() for p in college_text.split("\n") if p.strip()]
        college = parts[0] if parts else ""
        location = parts[1] if len(parts) > 1 else ""

        out.append(
            {
                "type": decision_type,
                "result": result,
                "college": college,
                "location": location,
            }
        )
    return out


def parse_student_row(tr: Tag) -> dict | None:
    cells = tr.find_all("td", recursive=False)
    if len(cells) != 5:
        return None

    name = cells[0].get_text(strip=True)
    if not name:
        return None

    gpa_text = cells[1].get_text(strip=True)
    gpa: float | None
    try:
        gpa = float(gpa_text) if gpa_text else None
    except ValueError:
        gpa = None

    return {
        "name": name,
        "gpa": gpa,
        "test_scores": parse_test_scores(cells[2]),
        "biography": parse_biography(cells[3]),
        "decisions": parse_decisions(cells[4]),
    }


def parse_html_file(path: Path) -> list[dict]:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    # The main student table is the only top-level <table class="table">
    # whose <thead> has a "Name" header. Inner test-score and decisions
    # tables are class="table table-sm" — we filter them out by header.
    students: list[dict] = []
    for table in soup.find_all("table", class_="table"):
        if "table-sm" in (table.get("class") or []):
            continue
        headers = [th.get_text(strip=True) for th in table.select("thead th")]
        if "Name" not in headers:
            continue
        tbody = table.find("tbody")
        if tbody is None:
            continue
        for tr in tbody.find_all("tr", class_="d-flex", recursive=False):
            row = parse_student_row(tr)
            if row is not None:
                row["_source_file"] = path.name
                students.append(row)
    return students


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    if not HTML_DIR.exists():
        raise SystemExit(f"Expected folder '{HTML_DIR}/' does not exist.")

    html_files = sorted(p for p in HTML_DIR.iterdir() if p.suffix.lower() == ".html")
    if not html_files:
        raise SystemExit(
            f"No .html files in '{HTML_DIR}/'. "
            "Save each destinations page there (right-click → Save As → 'Webpage, Complete')."
        )

    print(f"Parsing {len(html_files)} HTML file(s) from {HTML_DIR}/")
    all_students: list[dict] = []
    for path in tqdm(html_files, desc="Parsing"):
        try:
            rows = parse_html_file(path)
        except Exception as e:
            print(f"  Failed to parse {path.name}: {e}")
            continue
        all_students.extend(rows)
        print(f"  {path.name}: {len(rows)} student(s)")

    if not all_students:
        raise SystemExit("No students parsed. Inspect the HTML structure.")

    df = pd.DataFrame(all_students)

    # Drop exact duplicates by name (saving the same page twice shouldn't double-count).
    before = len(df)
    df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate student(s) by name.")

    # JSON-encode complex columns so the CSV stays clean and lossless.
    df["test_scores"] = df["test_scores"].apply(json.dumps)
    df["decisions"] = df["decisions"].apply(json.dumps)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} student rows to {OUT_CSV}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
