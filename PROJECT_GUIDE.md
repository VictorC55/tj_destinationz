# TJ Destinations — Custom LLM Project Guide

A step-by-step plan for scraping the TJHSST class-of-2026 destinations site, building a domain-tailored LLM, and comparing it against a general-purpose model (e.g., ChatGPT).

---

## 0. Before You Start — Ethics, Privacy, Legality

The destinations page contains self-reported data about real students (names, GPAs, test scores, college outcomes). Treat this as sensitive even though it is public.

- **Check `robots.txt` and the site's terms.** Visit `https://tjhsst26.sites.tjhsst.edu/robots.txt` and confirm the path you want to scrape is allowed. Throttle requests (1 per 2–5 seconds is polite).
- **Anonymize before modeling.** Replace each student name with a pseudonymous ID (`student_0001`, etc.) the moment the data leaves the scraper. The LLM should never see the name. Keep the name↔ID mapping in a separate file that is *not* indexed or sent to any API.
- **Get sign-off.** If this is for a class or research project, talk to the school administration (and ideally the students whose data you'll use) before publishing results or sharing the model.
- **Don't build a predictor that profiles individuals.** Aggregate questions ("what GPA range typically gets into UVA?") are reasonable. Individual questions ("will student X get in?") are not.

The rest of this guide assumes you'll anonymize the data before indexing.

---

## 1. Prerequisites

- Python 3.10+
- `pip` or `uv` for installing packages
- An OpenAI API key (or Anthropic, or local Ollama for an offline option)
- ~2 GB free disk space for embeddings + a local vector DB

Set environment variables in a `.env` file (don't commit it):

```bash
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 2. Libraries to Install

```bash
pip install \
  requests \
  beautifulsoup4 \
  lxml \
  pandas \
  python-dotenv \
  tqdm \
  openai \
  anthropic \
  chromadb \
  sentence-transformers \
  tiktoken \
  scikit-learn \
  matplotlib
```

What each is for:

| Library | Purpose |
|---|---|
| `requests`, `beautifulsoup4`, `lxml` | Scraping + HTML parsing |
| `pandas` | Cleaning the tabular destinations data |
| `python-dotenv` | Loading API keys from `.env` |
| `tqdm` | Progress bars during scraping/embedding |
| `openai` / `anthropic` | LLM API clients (your "general" model + generation in RAG) |
| `chromadb` | Local vector database for retrieval |
| `sentence-transformers` | Local embedding model (no API cost) |
| `tiktoken` | Counting tokens before API calls |
| `scikit-learn`, `matplotlib` | Evaluation metrics + plots |

Optional (if you go the fine-tuning route in Section 6):

```bash
pip install transformers datasets peft accelerate bitsandbytes torch
```

---

## 3. Step 1 — Scrape the Destinations Page

Inspect the page first (in your browser, View Source) to see whether the data is in a single HTML `<table>`, multiple tables, or paginated. The code below assumes one or more `<table>` elements; adjust selectors after you've looked.

Create `scrape_destinations.py`:

```python
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

URL = "https://tjhsst26.sites.tjhsst.edu/destinations/"
HEADERS = {
    "User-Agent": "TJ-Destinations-Research/0.1 (educational project; contact: you@school.edu)"
}
OUT = Path("data/raw_destinations.csv")
OUT.parent.mkdir(exist_ok=True)

def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text

def parse_tables(html: str) -> pd.DataFrame:
    # pandas.read_html handles most well-formed <table> elements
    tables = pd.read_html(html, flavor="lxml")
    if not tables:
        raise RuntimeError("No tables found — inspect the page and use BeautifulSoup directly.")
    # If there are multiple, concatenate. Often the destinations page is one big table.
    df = pd.concat(tables, ignore_index=True)
    return df

def parse_with_bs4(html: str) -> pd.DataFrame:
    """Fallback if the data isn't in a clean <table>."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    # EXAMPLE — replace selectors with whatever the page actually uses:
    for card in soup.select(".student-entry"):
        rows.append({
            "name":    card.select_one(".name").get_text(strip=True),
            "gpa":     card.select_one(".gpa").get_text(strip=True),
            "sat":     card.select_one(".sat").get_text(strip=True),
            "act":     card.select_one(".act").get_text(strip=True),
            "ecs":     card.select_one(".ecs").get_text(strip=True),
            "results": card.select_one(".college-results").get_text(strip=True),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    html = fetch(URL)
    time.sleep(2)  # polite throttle
    try:
        df = parse_tables(html)
    except Exception:
        df = parse_with_bs4(html)
    print(f"Scraped {len(df)} rows; columns: {list(df.columns)}")
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
```

Run it once, then **stop and inspect the CSV** — every site is different, and you'll likely need to tweak the selectors or split combined columns (e.g., "Accepted / Rejected / Waitlisted").

---

## 4. Step 2 — Clean and Anonymize

Create `clean_anonymize.py`:

```python
import hashlib
import pandas as pd
from pathlib import Path

RAW = Path("data/raw_destinations.csv")
CLEAN = Path("data/clean_destinations.csv")
NAME_MAP = Path("data/name_map.csv")  # KEEP THIS FILE PRIVATE

df = pd.read_csv(RAW)

# 1. Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 2. Pseudonymize names (deterministic so re-runs are stable)
def pseudo(name: str) -> str:
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"student_{h}"

df["student_id"] = df["name"].apply(pseudo)
name_map = df[["student_id", "name"]].drop_duplicates()
name_map.to_csv(NAME_MAP, index=False)  # do not commit
df = df.drop(columns=["name"])

# 3. Type coercion / cleanup
df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")
df["sat"] = pd.to_numeric(df["sat"], errors="coerce")
df["act"] = pd.to_numeric(df["act"], errors="coerce")

# 4. Split college results if it's a single string like "Accepted: A, B; Rejected: C"
def parse_results(text):
    out = {"accepted": [], "rejected": [], "waitlisted": [], "attending": None}
    if not isinstance(text, str):
        return out
    # adapt to whatever format the page uses
    for chunk in text.split(";"):
        if ":" in chunk:
            label, schools = chunk.split(":", 1)
            label = label.strip().lower()
            schools_list = [s.strip() for s in schools.split(",") if s.strip()]
            if label in out and isinstance(out[label], list):
                out[label] = schools_list
    return out

parsed = df["results"].apply(parse_results)
df["accepted"]   = parsed.apply(lambda d: d["accepted"])
df["rejected"]   = parsed.apply(lambda d: d["rejected"])
df["waitlisted"] = parsed.apply(lambda d: d["waitlisted"])

df.to_csv(CLEAN, index=False)
print(f"Wrote {CLEAN} ({len(df)} rows)")
```

Add `data/name_map.csv` to `.gitignore`.

---

## 5. Step 3 — Build the Custom LLM (RAG approach, recommended)

**Why RAG and not fine-tuning?** With a few hundred student records, fine-tuning won't teach the model the facts reliably — it'll learn style, not data. RAG (Retrieval Augmented Generation) actually retrieves the relevant rows at query time and feeds them to the model as context. It's faster, cheaper, easier to update, and much more accurate for this use case.

### 5a. Convert each student to a passage

Create `build_index.py`:

```python
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import ast

df = pd.read_csv("data/clean_destinations.csv")

def row_to_passage(row) -> str:
    accepted   = ", ".join(ast.literal_eval(row["accepted"]))   if isinstance(row["accepted"], str)   else ""
    rejected   = ", ".join(ast.literal_eval(row["rejected"]))   if isinstance(row["rejected"], str)   else ""
    waitlisted = ", ".join(ast.literal_eval(row["waitlisted"])) if isinstance(row["waitlisted"], str) else ""
    return (
        f"Student {row['student_id']} (TJHSST class of 2026). "
        f"GPA: {row.get('gpa', 'N/A')}. SAT: {row.get('sat', 'N/A')}. ACT: {row.get('act', 'N/A')}. "
        f"Extracurriculars: {row.get('ecs', 'N/A')}. "
        f"Accepted: {accepted or 'none listed'}. "
        f"Rejected: {rejected or 'none listed'}. "
        f"Waitlisted: {waitlisted or 'none listed'}."
    )

passages = [row_to_passage(r) for _, r in df.iterrows()]

# Local, free embedding model — no API cost
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name="tj_destinations",
    embedding_function=embedder,
)
collection.add(
    ids=df["student_id"].tolist(),
    documents=passages,
    metadatas=df[["gpa", "sat", "act"]].fillna(-1).to_dict(orient="records"),
)
print(f"Indexed {len(passages)} student passages.")
```

### 5b. Query layer

Create `rag_query.py`:

```python
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()
client_oai = OpenAI()  # uses OPENAI_API_KEY

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma = chromadb.PersistentClient(path="data/chroma")
collection = chroma.get_collection("tj_destinations", embedding_function=embedder)

SYSTEM = (
    "You are an assistant that answers questions about TJHSST class of 2026 college "
    "destinations using ONLY the provided context passages. If the answer isn't in the "
    "context, say so. Do not invent students, schools, or statistics. Refer to students "
    "by their pseudonymous ID."
)

def ask(question: str, k: int = 8) -> str:
    hits = collection.query(query_texts=[question], n_results=k)
    context = "\n\n".join(hits["documents"][0])
    resp = client_oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(ask("What GPA range typically got into UVA?"))
```

You now have a "custom LLM" — really a general LLM grounded in TJ-specific retrieval. For a class demo this is the right tool.

---

## 6. Optional Extension — Fine-Tuning a Small Open Model

Only worth doing if you want to demonstrate the fine-tuning workflow itself. It will *not* outperform RAG on factual recall for this dataset.

1. **Generate Q&A pairs** from the cleaned CSV (e.g., "What did student_abc1234 apply to?" → answer). Aim for ~1,000+ pairs by templating questions.
2. **Pick a base model.** `meta-llama/Llama-3.2-3B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3`.
3. **LoRA fine-tune** with `peft` + `transformers` + `accelerate` on a single GPU (Colab Pro works for 3B; 7B needs more).
4. **Serve locally** with `transformers` or convert to GGUF for `ollama`.

Skeleton:

```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1. Load + tokenize your Q&A pairs (qa_pairs.jsonl with {"prompt": ..., "completion": ...})
# 2. Wrap base model with LoRA adapters
# 3. Trainer(...).train()
# 4. model.save_pretrained("models/tj_lora")
```

---

## 7. Step 4 — Compare Custom vs. General LLM

### 7a. Build an evaluation set

Write 30–50 questions covering:

- **Factual lookup** ("List the schools student_X was accepted to.")
- **Aggregate stats** ("What's the median SAT of students who got into MIT?")
- **Pattern questions** ("What EC profiles correlate with Ivy acceptances?")
- **Out-of-scope** ("Who is the TJHSST principal?") — both should refuse or say they don't know.

Store as `eval/questions.jsonl`:

```json
{"id": "q1", "type": "factual", "question": "...", "reference_answer": "..."}
```

### 7b. Run both models

```python
import json
from rag_query import ask as ask_custom
from openai import OpenAI
client_oai = OpenAI()

def ask_general(q: str) -> str:
    resp = client_oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": q}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

results = []
for line in open("eval/questions.jsonl"):
    item = json.loads(line)
    results.append({
        **item,
        "custom_answer":  ask_custom(item["question"]),
        "general_answer": ask_general(item["question"]),
    })
json.dump(results, open("eval/results.json", "w"), indent=2)
```

### 7c. Score the results

Three complementary methods:

1. **Exact / fuzzy match** against reference answers for factual questions (`rapidfuzz` or a simple substring check).
2. **LLM-as-judge.** Have a third model (e.g., Claude) score each pair on a 1–5 rubric: factual accuracy, specificity, refusal-when-appropriate. Save prompts so the rubric is reproducible.
3. **Human review.** For the pattern questions, you (or students) rate which answer is more useful.

Plot per-category scores with `matplotlib`. Expected result: the custom RAG system massively wins on factual + aggregate questions; the general LLM may tie or win on out-of-scope refusals and general advice.

---

## 8. Project Deliverables Checklist

- [ ] `data/clean_destinations.csv` (anonymized)
- [ ] `data/name_map.csv` (private, gitignored)
- [ ] `data/chroma/` (vector index)
- [ ] `scrape_destinations.py`, `clean_anonymize.py`, `build_index.py`, `rag_query.py`
- [ ] `eval/questions.jsonl` + `eval/results.json`
- [ ] `report.md` with method, results table, and a discussion of where the custom system wins/loses
- [ ] A short section on limitations: dataset size, self-reported accuracy, single-school scope, anonymization tradeoffs

---

## 9. Suggested Timeline (for a class project)

| Week | Milestone |
|---|---|
| 1 | Scrape + inspect data, finalize anonymization |
| 2 | Build RAG index + query layer, sanity-check answers |
| 3 | Write eval questions, run both models, score |
| 4 | Write report, prepare demo |

---

## 10. Quick Reference — Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pipeline
python scrape_destinations.py
python clean_anonymize.py
python build_index.py

# interactive
python rag_query.py
```
