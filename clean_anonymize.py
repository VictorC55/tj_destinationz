"""
Clean and anonymize parsed student data.

Inputs:  data/raw_destinations.csv  (one row per student, real names)
Outputs:
    data/clean_destinations.csv  — anonymized, structured fields ready for RAG
    data/name_map.csv             — student_id -> real name (PRIVATE; gitignored)

The clean CSV is what feeds the RAG index. Real names live only in name_map.csv
and never reach the embedding model or the LLM.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw_destinations.csv")
CLEAN = Path("data/clean_destinations.csv")
NAME_MAP = Path("data/name_map.csv")


def pseudo_id(name: str) -> str:
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"student_{h}"


def safe_int(s) -> int | None:
    if s is None:
        return None
    try:
        return int(float(str(s).strip()))
    except (ValueError, TypeError):
        return None


def get_sat_total(scores: dict) -> int | None:
    total = safe_int(scores.get("SAT Total")) or safe_int(scores.get("SAT Composite"))
    if total is not None:
        return total
    math = safe_int(scores.get("SAT Math"))
    ebrw = (
        safe_int(scores.get("SAT EBRW"))
        or safe_int(scores.get("SAT Evidence-Based Reading and Writing"))
        or safe_int(scores.get("SAT Reading and Writing"))
    )
    if math is not None and ebrw is not None:
        return math + ebrw
    return None


def get_act_composite(scores: dict) -> int | None:
    return safe_int(scores.get("ACT Composite")) or safe_int(scores.get("ACT Total"))


def bucket_decisions(decisions: list) -> dict[str, list[str]]:
    """Split decisions by result. 'Accepted, attending' counts as both accepted and attending."""
    out: dict[str, list[str]] = {
        "accepted": [],
        "attending": [],
        "denied": [],
        "waitlisted": [],
        "deferred": [],
    }
    for d in decisions:
        college = (d.get("college") or "").strip()
        result = (d.get("result") or "").lower()
        if not college:
            continue
        if "attending" in result:
            out["accepted"].append(college)
            out["attending"].append(college)
        elif "accepted" in result:
            out["accepted"].append(college)
        elif "denied" in result or "rejected" in result:
            out["denied"].append(college)
        elif "waitlist" in result:
            out["waitlisted"].append(college)
        elif "deferred" in result:
            out["deferred"].append(college)
    return out


def redact_self_name(text: str, name: str) -> str:
    """Replace standalone occurrences of the student's first/last name with [REDACTED].

    Bios are usually first-person and don't self-reference by name, but this
    catches the cases where someone wrote "I, Jane Doe, ..." or signed off.
    """
    if not isinstance(text, str) or not text or not isinstance(name, str):
        return text if isinstance(text, str) else ""
    for part in re.split(r"\s+", name.strip()):
        if len(part) < 3:
            continue
        text = re.sub(rf"\b{re.escape(part)}\b", "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"{RAW} not found. Run parse_local_htmls.py first.")

    df = pd.read_csv(RAW)
    print(f"Loaded {len(df)} student row(s) from {RAW}")

    df["student_id"] = df["name"].apply(pseudo_id)

    # Persist the mapping BEFORE we strip names from the working frame.
    name_map = df[["student_id", "name"]].drop_duplicates().reset_index(drop=True)
    name_map.to_csv(NAME_MAP, index=False)
    print(f"Wrote name map ({len(name_map)} entries) to {NAME_MAP} — keep this private.")

    df["test_scores"] = df["test_scores"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else {}
    )
    df["decisions"] = df["decisions"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else []
    )

    df["sat_total"] = df["test_scores"].apply(get_sat_total)
    df["act_composite"] = df["test_scores"].apply(get_act_composite)

    buckets = df["decisions"].apply(bucket_decisions)
    df["accepted"] = buckets.apply(lambda b: b["accepted"])
    df["attending"] = buckets.apply(lambda b: b["attending"][0] if b["attending"] else None)
    df["denied"] = buckets.apply(lambda b: b["denied"])
    df["waitlisted"] = buckets.apply(lambda b: b["waitlisted"])
    df["deferred"] = buckets.apply(lambda b: b["deferred"])

    df["biography"] = df.apply(
        lambda row: redact_self_name(row["biography"], row["name"]),
        axis=1,
    )

    for col in ("test_scores", "decisions", "accepted", "denied", "waitlisted", "deferred"):
        df[col] = df[col].apply(json.dumps)

    out_cols = [
        "student_id",
        "gpa",
        "sat_total",
        "act_composite",
        "test_scores",
        "biography",
        "accepted",
        "attending",
        "denied",
        "waitlisted",
        "deferred",
        "decisions",
        "_source_file",
    ]
    df[out_cols].to_csv(CLEAN, index=False)
    print(f"Wrote {len(df)} anonymized row(s) to {CLEAN}")


if __name__ == "__main__":
    main()
