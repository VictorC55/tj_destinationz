"""
OpenAI-backed RAG query: retrieves student passages from ChromaDB, then asks
gpt-4o-mini to answer using only those passages.

Reads OPENAI_API_KEY from .env (see .env.example). Use:
    python rag_query.py "What GPA range typically got into UVA?"
or just `python rag_query.py` for an interactive prompt.

For a fully-local (no API) version, see rag_query_local.py.
"""

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "tj_destinations"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OPENAI_MODEL = "gpt-4o-mini"
N_RETRIEVE = 8
MAX_TOKENS = 600

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about TJHSST class of 2026 college "
    "destinations using ONLY the provided context passages. If the answer isn't in the "
    "context, say so. Do not invent students, schools, or statistics. Refer to students "
    "by their pseudonymous ID (e.g. student_dc4cbc69)."
)


def load_retriever():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME, embedding_function=embedder)


def load_openai() -> OpenAI:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI()


def retrieve(collection, question: str, k: int = N_RETRIEVE) -> str:
    hits = collection.query(query_texts=[question], n_results=k)
    return "\n\n---\n\n".join(hits["documents"][0])


def answer(client: OpenAI, collection, question: str) -> str:
    context = retrieve(collection, question)
    user_prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite specific student IDs."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    collection = load_retriever()
    client = load_openai()

    args = sys.argv[1:]
    if args:
        print(answer(client, collection, " ".join(args)))
        return

    print(f"OpenAI RAG ready ({OPENAI_MODEL}). Type a question (blank line to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        print()
        print(answer(client, collection, q))


if __name__ == "__main__":
    main()
