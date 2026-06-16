"""
Download a GGUF-format Llama 3.1 model into ./models/ for local inference.

Run once. The file lives entirely inside the project (no ~/.cache pollution)
so deleting the project folder fully removes it.

Default: Meta-Llama-3.1-8B-Instruct, 4-bit quantized (Q4_K_M) — ~4.6 GB.
Good quality/speed on Apple Silicon with 16 GB+ RAM.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LOCAL_DIR = Path("models")

def main() -> None:
    LOCAL_DIR.mkdir(exist_ok=True)
    target = LOCAL_DIR / FILENAME
    if target.exists():
        print(f"Already downloaded: {target} ({target.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading {FILENAME} from {REPO_ID} -> {LOCAL_DIR}/")
    print("(~4.6 GB; this can take several minutes depending on connection)")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
    )
    print(f"\nDone. Model at: {path}")


if __name__ == "__main__":
    main()
