from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = ROOT / "results"

PDF_PATH = DATA_DIR / "nist_ai_rmf.pdf"
PAGES_PATH = CACHE_DIR / "pages.json"
EXTRACTION_STATS_PATH = CACHE_DIR / "extraction_stats.json"
GOLD_QUESTIONS_PATH = EVAL_DIR / "gold_questions.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen-q4"
OLLAMA_TIMEOUT_SECONDS = 120
OLLAMA_NUM_PREDICT = 512

TOP_K = 5

# Token windows measured with the embedding-model tokenizer.
# Both stay under all-MiniLM-L6-v2 max_seq_length (256) after special tokens.
CHUNK_CONFIGS = {
    "A": {
        "label": "A",
        "chunk_size": 120,
        "overlap": 20,
        "reason": "Shorter, more specific chunks for pinpoint facts.",
    },
    "B": {
        "label": "B",
        "chunk_size": 220,
        "overlap": 40,
        "reason": "Longer chunks that keep more surrounding sentence context.",
    },
}


def chunks_path(config_id: str) -> Path:
    return CACHE_DIR / f"chunks_{config_id}.json"


def embeddings_path(config_id: str) -> Path:
    return CACHE_DIR / f"embeddings_{config_id}.npy"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
