"""
Side-by-side comparison UI: type a question, see both responses.

  Left  — Custom RAG (Llama 3.1 8B + retrieved TJ destinations passages)
  Right — Baseline  (same Llama 3.1 8B, no retrieval — model's own knowledge only)

The same model instance serves both queries so loading the GGUF only happens
once. Launch with:
    python compare_ui.py
and open the URL it prints (default http://127.0.0.1:7860).
"""

from pathlib import Path

import chromadb
import gradio as gr
from chromadb.utils import embedding_functions
from llama_cpp import Llama

from filters import extract_metadata_filter

MODEL_PATH = Path("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "tj_destinations"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

N_CTX = 8192
N_GPU_LAYERS = -1
MAX_TOKENS = 600
N_RETRIEVE = 4

RAG_SYSTEM = (
    "You are an assistant that answers questions about TJHSST class of 2026 college "
    "destinations using ONLY the provided context passages. If the answer isn't in the "
    "context, say so. Do not invent students, schools, or statistics. Refer to students "
    "by their pseudonymous ID (e.g. student_dc4cbc69). Attribute each fact to a specific "
    "student ID — do not merge data across students."
)

BASELINE_SYSTEM = (
    "You are a knowledgeable college admissions advisor. You answer questions from "
    "high school students about applying to college. You do not have access to data "
    "on any specific school's class results or individual student profiles. If you "
    "do not know an answer, say so plainly rather than inventing statistics or names."
)


print("Loading embedding model + Chroma collection...")
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_collection(COLLECTION_NAME, embedding_function=embedder)

if not MODEL_PATH.exists():
    raise SystemExit(f"{MODEL_PATH} not found. Run `python download_model.py` first.")

print(f"Loading {MODEL_PATH.name} (Metal)...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)
print("Model loaded. Starting UI...\n")


def retrieve(question: str) -> list[str]:
    where = extract_metadata_filter(question)
    kwargs = {"query_texts": [question], "n_results": N_RETRIEVE}
    if where is not None:
        kwargs["where"] = where
    hits = collection.query(**kwargs)
    docs = hits["documents"][0]
    if not docs and where is not None:
        hits = collection.query(query_texts=[question], n_results=N_RETRIEVE)
        docs = hits["documents"][0]
    return docs


def generate(system: str, user: str) -> str:
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    return out["choices"][0]["message"]["content"].strip()


def answer_both(question: str) -> tuple[str, str, str]:
    question = (question or "").strip()
    if not question:
        return "", "", ""

    passages = retrieve(question)
    rag_user = (
        f"Context passages:\n" + "\n\n---\n\n".join(passages)
        + f"\n\nQuestion: {question}\n\n"
        "Answer using only the context above. Cite specific student IDs."
    )
    rag_answer = generate(RAG_SYSTEM, rag_user)
    baseline_answer = generate(BASELINE_SYSTEM, question)

    retrieved_view = "\n\n────────\n\n".join(
        f"[{i + 1}] {p}" for i, p in enumerate(passages)
    )
    return rag_answer, baseline_answer, retrieved_view


with gr.Blocks(title="TJ Destinations: RAG vs Baseline") as demo:
    gr.Markdown(
        "# TJ Destinations — Custom RAG vs Baseline\n"
        "Same model (Llama 3.1 8B), same hardware. Only the **context** differs: "
        "the left side gets retrieved student passages; the right side sees nothing extra."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. What kinds of extracurriculars did students attending UVA have?",
        lines=2,
    )
    submit = gr.Button("Ask both", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Custom RAG (Llama 8B + retrieved TJ passages)")
            rag_out = gr.Textbox(label="", lines=18)
        with gr.Column():
            gr.Markdown("### Baseline (Llama 8B alone)")
            base_out = gr.Textbox(label="", lines=18)
    with gr.Accordion("Retrieved passages (what the RAG side saw)", open=False):
        retrieved_out = gr.Textbox(label="", lines=20)

    submit.click(
        fn=answer_both,
        inputs=question,
        outputs=[rag_out, base_out, retrieved_out],
    )
    question.submit(
        fn=answer_both,
        inputs=question,
        outputs=[rag_out, base_out, retrieved_out],
    )

if __name__ == "__main__":
    demo.launch()
