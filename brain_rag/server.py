"""MCP server exposing hybrid RAG search over the Obsidian Brain-vault."""

import threading

from mcp.server.fastmcp import FastMCP

from . import indexer, search
from .config import MODEL_NAME, VAULT

mcp = FastMCP("brain-rag")


@mcp.tool()
def brain_search(query: str, limit: int = 8) -> list[dict]:
    """Semantic + keyword hybrid search over the Obsidian Brain-vault.

    Returns the most relevant note chunks (path, heading, text, score).
    Use natural-language queries in French or English.
    """
    return search.search(query, limit=min(max(1, limit), 30))


@mcp.tool()
def brain_get_note(path: str) -> str:
    """Read the full content of a vault note by its relative path
    (as returned in brain_search results)."""
    return search.read_note(path)


@mcp.tool()
def brain_reindex(full: bool = False) -> dict:
    """Re-index the vault. Incremental by default (only changed notes);
    full=True rebuilds everything."""
    return indexer.reindex(full=full)


@mcp.tool()
def brain_status() -> dict:
    """Index statistics: chunk count, files, vault path, embedding model."""
    table = search.get_table()
    return {
        "vault": str(VAULT),
        "model": MODEL_NAME,
        "chunks": table.count_rows(),
    }


def main():
    # Catch up on vault changes since last run, without blocking startup.
    threading.Thread(target=lambda: _safe_reindex(), daemon=True).start()
    mcp.run()


def _safe_reindex():
    try:
        indexer.reindex()
    except Exception:
        pass


if __name__ == "__main__":
    main()
