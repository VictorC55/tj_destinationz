"""
Heuristic extraction of numeric metadata filters from a natural-language question.

Turns phrases like "GPA over 4.2" or "SAT below 1500" into ChromaDB where-clauses,
so vector search runs only over the matching subset of students. Returns None when
no filter is detected, which means "search everything semantically".

Only handles GPA, SAT total, and ACT composite — the three numeric fields stored
in the Chroma collection's metadata. Extending to other fields means adding rows
to FIELDS below.
"""

import re

FIELDS = [
    ("gpa", [r"gpa"]),
    ("sat_total", [r"sat\s*(?:total|composite|score)?"]),
    ("act_composite", [r"act\s*(?:composite|total|score)?"]),
]

GE_OPS = r"(?:>=?|over|above|at\s*least|greater\s*than|higher\s*than|more\s*than)"
LE_OPS = r"(?:<=?|under|below|at\s*most|less\s*than|lower\s*than|fewer\s*than)"


def extract_metadata_filter(question: str) -> dict | None:
    if not question:
        return None
    q = question.lower()
    clauses: list[dict] = []

    for field, keywords in FIELDS:
        for kw in keywords:
            ge = re.search(rf"\b{kw}\s*{GE_OPS}\s*(\d+(?:\.\d+)?)", q)
            if ge:
                clauses.append({field: {"$gte": float(ge.group(1))}})
                break
            le = re.search(rf"\b{kw}\s*{LE_OPS}\s*(\d+(?:\.\d+)?)", q)
            if le:
                clauses.append({field: {"$lte": float(le.group(1))}})
                break

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
