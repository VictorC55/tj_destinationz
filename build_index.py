"""
Embed the anonymized student data into a local ChromaDB index for RAG retrieval.

For each student, build one natural-language passage that summarizes GPA, test
scores, admission outcomes, and biography. Each passage is embedded with a
local sentence-transformers model (free, no API call) and stored in the
'tj_destinations' collection at data/chroma/.

Re-running this script wipes and rebuilds the collection so re-indexing is clean
after edits to the source data.
"""

import json
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from tqdm import tqdm

CLEAN = Path("data/clean_destinations.csv")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "tj_destinations"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim, stronger semantic matching than MiniLM
BATCH_SIZE = 32


def fmt_score(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def loads_list(v) -> list:
    if isinstance(v, str) and v:
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


def row_to_passage(row: pd.Series) -> str:
    accepted = loads_list(row["accepted"])
    denied = loads_list(row["denied"])
    waitlisted = loads_list(row["waitlisted"])
    deferred = loads_list(row["deferred"])

    raw_scores = row["test_scores"]
    if isinstance(raw_scores, str) and raw_scores:
        try:
            scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            scores = {}
    else:
        scores = {}

    attending = row.get("attending")
    if not isinstance(attending, str) or pd.isna(attending):
        attending = None

    biography = row.get("biography") or ""
    if not isinstance(biography, str):
        biography = ""
    biography = biography.strip() or "(no biography provided)"

    score_str = ", ".join(f"{k} {v}" for k, v in scores.items()) or "none reported"

    # Front-load a one-line summary — embeddings weight the beginning of a passage
    # more heavily, so concentrating the most distinguishing facts up front improves
    # semantic recall on queries like "students attending UF with high GPA".
    summary = (
        f"Profile: GPA {fmt_score(row.get('gpa'))}, "
        f"SAT {fmt_score(row.get('sat_total'))}, "
        f"ACT {fmt_score(row.get('act_composite'))}, "
        f"attending {attending or 'undecided'}, "
        f"{len(accepted)} acceptance(s), {len(denied)} denial(s)."
    )

    lines = [
        f"Student {row['student_id']} (TJHSST class of 2026).",
        summary,
        f"Accepted to: {', '.join(accepted) if accepted else 'none listed'}.",
        f"Attending: {attending or 'not specified'}.",
        f"Denied from: {', '.join(denied) if denied else 'none listed'}.",
        f"Waitlisted at: {', '.join(waitlisted) if waitlisted else 'none listed'}.",
        f"Deferred at: {', '.join(deferred) if deferred else 'none listed'}.",
        f"All test scores: {score_str}.",
        f"Biography: {biography}",
    ]
    return "\n".join(lines)


def to_metadata(row: pd.Series) -> dict:
    """ChromaDB metadata values must be str/int/float/bool (no None, no NaN, no lists)."""
    meta: dict = {}
    for key in ("gpa", "sat_total", "act_composite"):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            meta[key] = float(v)
    attending = row.get("attending")
    if isinstance(attending, str) and attending and not pd.isna(attending):
        meta["attending"] = attending
    return meta


def main() -> None:
    if not CLEAN.exists():
        raise SystemExit(f"{CLEAN} not found. Run clean_anonymize.py first.")

    df = pd.read_csv(CLEAN)
    print(f"Loaded {len(df)} student row(s) from {CLEAN}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Replace the collection on every run so re-indexing is clean.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~80MB)...")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    ids = df["student_id"].tolist()
    passages = [row_to_passage(row) for _, row in df.iterrows()]
    metadatas = [to_metadata(row) for _, row in df.iterrows()]
    # ChromaDB rejects empty-dict metadata in some versions — substitute a tiny sentinel.
    metadatas = [m if m else {"_": ""} for m in metadatas]

    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Embedding"):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            documents=passages[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"\nIndexed {collection.count()} students into '{COLLECTION_NAME}' at {CHROMA_DIR}/")
    print("\n--- Sample passage ---")
    print(passages[0][:900] + ("..." if len(passages[0]) > 900 else ""))


if __name__ == "__main__":
    main()
