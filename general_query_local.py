"""
No-RAG baseline: same Llama 3.1 8B model as rag_query_local.py, but without
the ChromaDB retrieval step. Use this as the control in an A/B comparison
to isolate what the RAG retrieval layer actually contributes.

Same model, same hardware, same generation settings — only the context differs.

Use:
    python general_query_local.py "What GPA range typically got into UVA?"
or just `python general_query_local.py` for interactive mode.
"""

import sys
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = Path("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

N_CTX = 8192
N_GPU_LAYERS = -1
MAX_TOKENS = 600

SYSTEM_PROMPT = (
    "You are a knowledgeable college admissions advisor. You answer questions "
    "from high school students about applying to college. You do not have "
    "access to data on any specific school's class results or any individual "
    "student's profile. If you do not know an answer, say so plainly rather "
    "than inventing statistics or student names."
)


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


def answer(llm: Llama, question: str) -> str:
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    return out["choices"][0]["message"]["content"].strip()


def main() -> None:
    llm = load_llm()

    args = sys.argv[1:]
    if args:
        print(answer(llm, " ".join(args)))
        return

    print("No-RAG baseline ready. Type a question (blank line to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        print()
        print(answer(llm, q))


if __name__ == "__main__":
    main()
