"""
Side-by-side comparison UI: type a question, see both responses.

  Left  — Custom RAG (Llama 3.1 8B + retrieved TJ destinations passages)
  Right — Baseline  (same Llama 3.1 8B, no retrieval — model's own knowledge only)

The same model instance serves both queries so loading the GGUF only happens
once. Launch with:
    python compare_ui.py
and open the URL it prints (default http://127.0.0.1:7860).
"""

import time
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
N_RETRIEVE = 6

RAG_SYSTEM = (
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


def stream_generate(system: str, user: str):
    """Yield text deltas as the model produces them."""
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            yield delta


def answer_both(question: str):
    """Generator: yields (rag_text, baseline_text, retrieved_view, status) on every new token."""
    question = (question or "").strip()
    if not question:
        yield "", "", "", ""
        return

    t0 = time.time()
    rag_done_at: float | None = None

    def status(phase: str) -> str:
        elapsed = time.time() - t0
        if phase == "rag":
            return f"⏱ {elapsed:.1f}s — generating RAG response…"
        if phase == "baseline":
            rag_t = rag_done_at - t0 if rag_done_at else 0
            base_t = elapsed - (rag_done_at - t0 if rag_done_at else 0)
            return f"⏱ {elapsed:.1f}s — RAG done in {rag_t:.1f}s, baseline {base_t:.1f}s and counting…"
        if phase == "done":
            rag_t = rag_done_at - t0 if rag_done_at else 0
            base_t = elapsed - (rag_done_at - t0 if rag_done_at else 0)
            return f"✅ Done in {elapsed:.1f}s — RAG: {rag_t:.1f}s, baseline: {base_t:.1f}s"
        return f"⏱ {elapsed:.1f}s"

    passages = retrieve(question)
    retrieved_view = "\n\n────────\n\n".join(
        f"[{i + 1}] {p}" for i, p in enumerate(passages)
    )
    rag_user = (
        f"Context passages:\n" + "\n\n---\n\n".join(passages)
        + f"\n\nQuestion: {question}\n\n"
        "Answer using only the context above. When recommending activities, programs, "
        "internships, or clubs, name specific ones that students in the context actually "
        "did and cite which student (by ID) did each one. If the context lacks specifics "
        "relevant to this question, say so rather than offering generic advice."
    )

    rag_text = ""
    base_text = ""
    yield "⏳ Generating…", "⏳ Waiting for RAG to finish…", retrieved_view, status("rag")

    for token in stream_generate(RAG_SYSTEM, rag_user):
        rag_text += token
        yield rag_text, "⏳ Waiting for RAG to finish…", retrieved_view, status("rag")

    rag_done_at = time.time()
    for token in stream_generate(BASELINE_SYSTEM, question):
        base_text += token
        yield rag_text, base_text, retrieved_view, status("baseline")

    yield rag_text, base_text, retrieved_view, status("done")


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
    status_out = gr.Markdown("")
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
        outputs=[rag_out, base_out, retrieved_out, status_out],
    )
    question.submit(
        fn=answer_both,
        inputs=question,
        outputs=[rag_out, base_out, retrieved_out, status_out],
    )

if __name__ == "__main__":
    demo.launch()
