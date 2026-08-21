"""Hybrid search: vector (fastembed) + BM25 full-text, merged with reciprocal
rank fusion, then reranked with a cross-encoder for precision."""

import lancedb

from .config import DATA_DIR, RERANK_MODEL, SECRET_RE, TABLE_NAME, VAULT
from .indexer import get_model

_db = None
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def get_table():
    global _db
    if _db is None:
        _db = lancedb.connect(str(DATA_DIR))
    if TABLE_NAME not in _db.table_names():
        raise RuntimeError("Index absent — lance d'abord: brain-rag-index")
    return _db.open_table(TABLE_NAME)


def _rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for results in result_lists:
        for rank, row in enumerate(results):
            rid = row["id"]
            by_id.setdefault(rid, row)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for rid, score in ranked:
        row = dict(by_id[rid])
        row["score"] = round(score, 6)
        out.append(row)
    return out


COLUMNS = ["id", "path", "heading", "text"]


def search(query: str, limit: int = 8, rerank: bool = True) -> list[dict]:
    table = get_table()
    fetch = max(limit * 3, 15)

    vec = list(get_model().embed([query]))[0]
    vector_hits = (
        table.search(vec.tolist(), vector_column_name="vector")
        .select(COLUMNS)
        .limit(fetch)
        .to_list()
    )
    try:
        fts_hits = (
            table.search(query, query_type="fts").select(COLUMNS).limit(fetch).to_list()
        )
    except Exception:
        fts_hits = []

    def clean(rows):
        return [{c: r[c] for c in COLUMNS} for r in rows]

    fused = _rrf([clean(vector_hits), clean(fts_hits)])

    if not rerank or not fused:
        return fused[:limit]

    # Cross-encoder rerank the fused candidate pool: RRF orders by rank
    # agreement, the reranker scores actual query/text relevance. Biggest
    # precision lever on an already-hybrid pipeline (Anthropic: -67% failures).
    candidates = fused[: max(fetch, 20)]
    try:
        scores = list(get_reranker().rerank(query, [c["text"] for c in candidates]))
        for row, s in zip(candidates, scores):
            row["score"] = round(float(s), 6)
        candidates.sort(key=lambda r: r["score"], reverse=True)
        return candidates[:limit]
    except Exception:
        # ponytail: reranker is a quality boost, never a hard dependency —
        # fall back to the RRF order if the model can't load.
        return fused[:limit]


def read_note(rel_path: str) -> str:
    rel = rel_path.strip().lstrip("/")
    abs_path = (VAULT / rel).resolve()
    if not str(abs_path).startswith(str(VAULT) + "/"):
        raise PermissionError("Path escapes vault")
    if SECRET_RE.search(rel) or ".git" in rel.split("/") or "secure-docs" in rel.split("/"):
        raise PermissionError("Path blocked")
    return abs_path.read_text(encoding="utf-8", errors="replace")
