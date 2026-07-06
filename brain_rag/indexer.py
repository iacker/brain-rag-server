"""Incremental indexer: Brain-vault markdown -> chunks -> embeddings -> LanceDB."""

import re
import sys

import lancedb

from .config import (
    CHUNK_OVERLAP,
    DATA_DIR,
    EXCLUDED_DIRS,
    MAX_CHUNK_CHARS,
    MODEL_NAME,
    SECRET_RE,
    TABLE_NAME,
    VAULT,
)

_model = None


def get_model():
    global _model
    if _model is None:
        import os
        from pathlib import Path

        # Pin the model cache to a stable dir (fastembed defaults to the system
        # tmp dir, which macOS purges), and skip HuggingFace Hub checks entirely
        # once the model is on disk so cold starts work offline.
        cache_dir = Path(
            os.environ.setdefault(
                "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_stem = MODEL_NAME.split("/")[-1]
        if any(cache_dir.glob(f"models--*{model_stem}*")):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def iter_vault_files():
    for path in sorted(VAULT.rglob("*.md")):
        rel = path.relative_to(VAULT).as_posix()
        parts = rel.split("/")
        if any(p in EXCLUDED_DIRS for p in parts):
            continue
        if SECRET_RE.search(rel):
            continue
        yield rel, path


def chunk_note(rel: str, text: str):
    """Split on markdown headings, then hard-split oversized sections."""
    sections = []
    heading = ""
    buf = []
    for line in text.splitlines():
        if re.match(r"^#{1,4}\s", line):
            if buf and "".join(buf).strip():
                sections.append((heading, "\n".join(buf)))
            heading = line.lstrip("#").strip()
            buf = [line]
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((heading, "\n".join(buf)))

    chunks = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        start = 0
        while start < len(body):
            piece = body[start : start + MAX_CHUNK_CHARS]
            chunks.append((heading, piece))
            if start + MAX_CHUNK_CHARS >= len(body):
                break
            start += MAX_CHUNK_CHARS - CHUNK_OVERLAP
    return [
        {
            "id": f"{rel}#{i}",
            "path": rel,
            "heading": heading,
            "text": piece,
        }
        for i, (heading, piece) in enumerate(chunks)
    ]


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def reindex(full: bool = False) -> dict:
    """Incremental by default: only changed/new/deleted files are touched.

    Guarded by a file lock: Claude Code and Hermes each spawn their own server
    process, and concurrent reindexes corrupt or duplicate the table.
    """
    import fcntl

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = open(DATA_DIR / ".reindex.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return {"skipped": "reindex already running in another process"}
    try:
        return _reindex_locked(full)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _reindex_locked(full: bool) -> dict:
    db = lancedb.connect(str(DATA_DIR))

    existing_mtimes: dict[str, float] = {}
    table = None
    if TABLE_NAME in db.table_names() and not full:
        table = db.open_table(TABLE_NAME)
        arrow = table.to_arrow().select(["path", "mtime"])
        for batch in arrow.to_batches():
            for p, m in zip(
                batch.column("path").to_pylist(), batch.column("mtime").to_pylist()
            ):
                existing_mtimes[p] = max(m, existing_mtimes.get(p, 0.0))
    elif TABLE_NAME in db.table_names() and full:
        db.drop_table(TABLE_NAME)

    on_disk: dict[str, float] = {}
    changed: list[tuple[str, object]] = []
    for rel, path in iter_vault_files():
        mtime = path.stat().st_mtime
        on_disk[rel] = mtime
        if existing_mtimes.get(rel) != mtime:
            changed.append((rel, path))
    deleted = [p for p in existing_mtimes if p not in on_disk]

    if not changed and not deleted:
        return {"indexed_files": 0, "deleted_files": 0, "chunks": 0, "total_files": len(on_disk)}

    rows = []
    for rel, path in changed:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for chunk in chunk_note(rel, text):
            chunk["mtime"] = on_disk[rel]
            rows.append(chunk)

    if rows:
        model = get_model()
        texts = [f"{r['path']} {r['heading']}\n{r['text']}" for r in rows]
        vectors = list(model.embed(texts, batch_size=64))
        for row, vec in zip(rows, vectors):
            row["vector"] = [float(x) for x in vec]

    if table is not None:
        stale = [rel for rel, _ in changed if rel in existing_mtimes] + deleted
        for rel in stale:
            table.delete(f"path = '{_sql_quote(rel)}'")
        if rows:
            table.add(rows)
    elif rows:
        table = db.create_table(TABLE_NAME, data=rows)

    if table is not None:
        table.create_fts_index("text", replace=True, use_tantivy=False)

    return {
        "indexed_files": len(changed),
        "deleted_files": len(deleted),
        "chunks": len(rows),
        "total_files": len(on_disk),
    }


def main():
    full = "--full" in sys.argv
    stats = reindex(full=full)
    print(stats)


if __name__ == "__main__":
    main()
