import os
import re
from pathlib import Path

VAULT = Path(os.environ.get("OBSIDIAN_VAULT_PATH", str(Path.home() / "Brain-vault"))).resolve()
DATA_DIR = Path(os.environ.get("BRAIN_RAG_DB", str(Path.home() / "brain-rag" / "data")))
TABLE_NAME = "chunks"

# Multilingual (French + English notes), 384-dim, ONNX/CPU.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Second-stage cross-encoder reranker (FR+EN). Loaded lazily at query time only.
# Cuts retrieval failures ~a third on top of hybrid search (Anthropic/BEIR).
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# Mirrors neuromancer's safety rules, plus vault dirs that are not notes.
EXCLUDED_DIRS = {
    ".git", ".obsidian", ".omc", ".raw", ".trash",
    "secure-docs", "agents", "node_modules", "__pycache__",
}
SECRET_RE = re.compile(
    r"(^|/)(\.env[^/]*|.*secret.*|.*token.*|.*key.*|.*wallet.*|.*seed.*|.*auth.*)(\.|$|/)",
    re.IGNORECASE,
)

MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200
