"""
Ollama-backed RAG query. Same retrieval as rag_query_local.py but generates
with a model served by a local Ollama process instead of llama-cpp-python.

Use this on Windows machines (or anywhere else) where llama-cpp-python won't
build or run. Ollama bundles its own runtime with CPU/GPU detection, so it
sidesteps the build chain and AVX-instruction issues that come with the
prebuilt llama-cpp-python wheels.

Prereqs:
  1. Install Ollama from https://ollama.com/download (one-click installer)
  2. `ollama pull llama3.1:8b`
  3. Make sure `ollama serve` is running (the installer usually auto-starts it)

Use:
    python rag_query_ollama.py "What GPA range got into UVA?"
or interactive: `python rag_query_ollama.py`
"""

import os
import sys
from pathlib import Path

import chromadb
import requests
from chromadb.utils import embedding_functions

from filters import extract_metadata_filter

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "tj_destinations"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

N_RETRIEVE = 6
MAX_TOKENS = 600

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about TJHSST class of 2026 college "
    "destinations using ONLY the provided context passages. If the answer isn't in the "
    "context, say so. Do not invent students, schools, or statistics. Refer to students "
    "by their pseudonymous ID (e.g. student_dc4cbc69). Attribute each fact to a specific "
    "student ID — do not merge data across students.\n\n"
    "When giving advice or recommendations, you MUST cite specific named activities "
    "from the context — actual clubs, internships, research programs, summer programs, "
    "sports, jobs, or competitions that students in the context did. Concrete examples "
    "from the data are far more valuable than abstract advice. Do NOT give generic "
    "recommendations like 'join clubs that align with your interests' or 'pursue what "
    "you're passionate about' — instead name specific things, e.g. 'student_dc4cbc69 "
    "did ASSIP bio research at GMU and a Georgetown MedStar Hospital internship'. "
    "If the retrieved students don't have activities relevant to the question, say so "
    "plainly rather than substituting generic guidance."
)


def load_retriever():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME, embedding_function=embedder)


def retrieve(collection, question: str, k: int = N_RETRIEVE) -> str:
    where = extract_metadata_filter(question)
    kwargs = {"query_texts": [question], "n_results": k}
    if where is not None:
        kwargs["where"] = where
    hits = collection.query(**kwargs)
    docs = hits["documents"][0]
    if not docs and where is not None:
        hits = collection.query(query_texts=[question], n_results=k)
        docs = hits["documents"][0]
    return "\n\n---\n\n".join(docs)


def check_ollama() -> None:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running?\n"
            f"  Start it with `ollama serve`, then verify with `ollama list`.\n"
            f"  Underlying error: {e}"
        )
    available = {m["name"] for m in r.json().get("models", [])}
    if not any(name.startswith(OLLAMA_MODEL.split(":", 1)[0]) for name in available):
        raise SystemExit(
            f"Model '{OLLAMA_MODEL}' isn't pulled yet. Run:\n"
            f"  ollama pull {OLLAMA_MODEL}"
        )


def answer(collection, question: str) -> str:
    context = retrieve(collection, question)
    user_prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. When recommending activities, programs, "
        "internships, or clubs, name specific ones that students in the context actually "
        "did and cite which student (by ID) did each one. If the context lacks specifics "
        "relevant to this question, say so rather than offering generic advice."
    )
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": MAX_TOKENS},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def main() -> None:
    check_ollama()
    collection = load_retriever()

    args = sys.argv[1:]
    if args:
        print(answer(collection, " ".join(args)))
        return

    print(f"Ollama RAG ready ({OLLAMA_MODEL}). Type a question (blank line to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        print()
        print(answer(collection, q))


if __name__ == "__main__":
    main()
