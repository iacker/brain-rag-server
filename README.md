# brain-rag-server

Local, fully self-hosted hybrid-RAG [MCP](https://modelcontextprotocol.io) server for an Obsidian (or any markdown) vault. Plug your second brain into Claude Code, Hermes, or any MCP client — no cloud, no GPU, no database server, no daemon.

```
Obsidian vault (*.md)
   ↓  incremental indexer (mtime-based, file-locked)
fastembed — paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ONNX, CPU)
   ↓
LanceDB (embedded) — vector index + BM25 full-text index
   ↓  reciprocal rank fusion
MCP server (stdio)
   ├── Claude Code
   └── Hermes / any MCP client
```

## Why

- **Hybrid search**: semantic (multilingual embeddings) + keyword (BM25), merged with RRF — better than either alone, works in French and English.
- **Zero infra**: LanceDB is embedded (like SQLite), fastembed runs ONNX on CPU. Total footprint < 1 GB, no background processes.
- **Incremental**: only changed/new/deleted notes are re-embedded. A file lock makes concurrent server startups (multiple MCP clients) safe.
- **Safe by default**: never indexes `.git`, `secure-docs`, `agents/`, `.obsidian`, or secret-looking filenames (`*token*`, `*key*`, `*auth*`, `.env*`, …).

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/iacker/brain-rag-server
cd brain-rag-server
uv sync
```

## Index your vault

```bash
export OBSIDIAN_VAULT_PATH="$HOME/my-vault"   # default: ~/Brain-vault
uv run brain-rag-index --full
```

First run downloads the embedding model (~500 MB, cached) and embeds every note. Subsequent runs are incremental. The index lives in `data/` (override with `BRAIN_RAG_DB`).

## Register the MCP server

**Claude Code:**

```bash
claude mcp add --scope user brain-rag -- /path/to/brain-rag-server/bin/brain-rag-server
```

**Hermes** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  brain-rag:
    command: /path/to/brain-rag-server/bin/brain-rag-server
    enabled: true
    connect_timeout: 30
    timeout: 120
```

**Any MCP client** (generic `mcpServers` JSON):

```json
{
  "mcpServers": {
    "brain-rag": {
      "command": "/path/to/brain-rag-server/bin/brain-rag-server",
      "env": { "OBSIDIAN_VAULT_PATH": "/path/to/your/vault" }
    }
  }
}
```

The server catches up on vault changes at startup (non-blocking, lock-guarded).

## Tools

| Tool | Description |
|------|-------------|
| `brain_search(query, limit)` | Hybrid semantic + keyword search; returns chunks with path, heading, text, score |
| `brain_get_note(path)` | Read a full note by vault-relative path (path-traversal and secret-file safe) |
| `brain_reindex(full)` | Incremental reindex (or full rebuild with `full=true`) |
| `brain_status()` | Vault path, model, chunk count |

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `OBSIDIAN_VAULT_PATH` | `~/Brain-vault` | Vault to index |
| `BRAIN_RAG_DB` | `<project>/data` | LanceDB storage directory |
| `FASTEMBED_CACHE_PATH` | `~/.cache/fastembed` | Embedding model cache (pinned outside the system tmp dir so the OS never purges it) |

**Offline by design**: the HuggingFace Hub is only contacted once, to download the embedding model. As soon as the model is in the cache, `HF_HUB_OFFLINE=1` is set automatically — every later start works with no network at all.

Chunking (heading-aware, ~1800 chars, 200 overlap), the embedding model, and exclusion rules are constants in `brain_rag/config.py`.
