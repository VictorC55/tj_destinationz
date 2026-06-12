"""
Fully-local RAG query: retrieves student passages from ChromaDB, then generates
an answer with a quantized Llama 3.1 8B running on Metal via llama-cpp-python.

No API calls, no data leaves your machine. Use:
    python rag_query_local.py "What GPA range typically got into UVA?"
or just `python rag_query_local.py` for an interactive prompt.
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama

MODEL_PATH = Path("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "tj_destinations"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

N_RETRIEVE = 8
N_CTX = 8192        # context window the model sees (prompt + completion)
N_GPU_LAYERS = -1   # -1 = offload everything to Metal on Apple Silicon
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


def load_llm() -> Llama:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found. Run `python download_model.py` first."
        )
    print(f"Loading {MODEL_PATH.name} (Metal, all layers on GPU)...", file=sys.stderr)
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )


def retrieve(collection, question: str, k: int = N_RETRIEVE) -> str:
    hits = collection.query(query_texts=[question], n_results=k)
    return "\n\n---\n\n".join(hits["documents"][0])


def answer(llm: Llama, collection, question: str) -> str:
    context = retrieve(collection, question)
    user_prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite specific student IDs."
    )
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    return out["choices"][0]["message"]["content"].strip()


def main() -> None:
    collection = load_retriever()
    llm = load_llm()

    args = sys.argv[1:]
    if args:
        print(answer(llm, collection, " ".join(args)))
        return

    print("Local RAG ready. Type a question (blank line to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        print()
        print(answer(llm, collection, q))


if __name__ == "__main__":
    main()
