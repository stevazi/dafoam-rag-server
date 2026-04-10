#!/usr/bin/env python3
"""DAFoam Chroma RAG — persistent SSE MCP server.

Runs as a background service; all Copilot CLI / Continue.dev sessions
connect to it via SSE. The embedding model is loaded ONCE at startup.

Default port: 29310  (set CHROMA_MCP_PORT in .env to override)

Start:
    python src/mcp/chroma_sse_server.py
    python src/mcp/chroma_sse_server.py --port 29310

MCP config entry:
    "dafoam-rag": {
      "type": "sse",
      "url": "http://127.0.0.1:29310/sse"
    }

Exposed tools:
    search_codebase  — DAFoam Python + C++ source
    search_docs      — DAFoam documentation (dafoam.github.io)
    search_tests     — Test scripts + OpenFOAM case configs
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s dafoam-rag-sse -- %(message)s",
)
log = logging.getLogger(__name__)

import chromadb
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
import uvicorn

from config.settings import settings
from src.rag.embeddings import get_embed_model

# Singletons — loaded once at process startup
_embed_model = None
_code_collection: chromadb.Collection | None = None
_docs_collection: chromadb.Collection | None = None
_tests_collection: chromadb.Collection | None = None


def startup() -> None:
    global _embed_model, _code_collection, _docs_collection, _tests_collection

    log.info("Loading embedding model: %s", settings.embed_model)
    _embed_model = get_embed_model()
    log.info("Embedding model ready.")

    for attr, dir_cfg, col_cfg in [
        ("_code_collection",  settings.chroma_code_dir,  settings.chroma_collection_code),
        ("_docs_collection",  settings.chroma_docs_dir,  settings.chroma_collection_docs),
        ("_tests_collection", settings.chroma_tests_dir, settings.chroma_collection_tests),
    ]:
        db_path = str(_PROJECT_ROOT / dir_cfg)
        try:
            client = chromadb.PersistentClient(path=db_path)
            col = client.get_collection(col_cfg)
            globals()[attr] = col
            log.info("Collection '%s': %d chunks", col_cfg, col.count())
        except Exception as exc:
            log.warning("Collection '%s' unavailable: %s — run the matching index script.", col_cfg, exc)

    log.info("SSE server ready.")


# ── Search helper ─────────────────────────────────────────────────────────────

def _search(collection: chromadb.Collection, query: str, n_results: int) -> str:
    embeddings = _embed_model.get_text_embedding_batch([f"query: {query}"])

    results = collection.query(
        query_embeddings=embeddings,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas",  [[]])[0]
    dists = results.get("distances",  [[]])[0]
    if not docs:
        return "No results found."

    parts: list[str] = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = round(1 - dist, 3)
        file_path = meta.get("file_path") or meta.get("url") or meta.get("source", "?")
        doc_type  = meta.get("doc_type", "")
        label = f"{file_path}  [{doc_type}]" if doc_type else file_path
        parts.append(f"**{label}** (score: {score})\n```\n{doc.strip()}\n```")
    return "\n\n---\n\n".join(parts)


# ── MCP server ────────────────────────────────────────────────────────────────

server = Server("dafoam-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    code_n  = _code_collection.count()  if _code_collection  else 0
    docs_n  = _docs_collection.count()  if _docs_collection  else 0
    tests_n = _tests_collection.count() if _tests_collection else 0

    return [
        Tool(
            name="search_codebase",
            description=(
                f"Search the indexed DAFoam codebase ({code_n:,} chunks — Python + C++ source). "
                "Returns relevant code snippets with file paths and relevance scores. "
                "Use to find DAFoam classes, methods, solver logic, or adjoint implementations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "Natural language or code search query"},
                    "n_results": {"type": "integer", "description": "Results to return (default 5, max 20)", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_docs",
            description=(
                f"Search the indexed DAFoam documentation ({docs_n:,} chunks from dafoam.github.io). "
                "Use for installation guides, tutorials, API references, solver descriptions, "
                "and configuration options."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "Natural language search query"},
                    "n_results": {"type": "integer", "description": "Results to return (default 5, max 20)", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_tests",
            description=(
                f"Search indexed DAFoam test cases and OpenFOAM case configs ({tests_n:,} chunks). "
                "Use to find example DAOPTION configurations, solver setups, and reference cases "
                "for aerodynamic, thermal, structural, or turbomachinery optimization."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "Natural language or solver/option name query"},
                    "n_results": {"type": "integer", "description": "Results to return (default 5, max 20)", "default": 5},
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        query = arguments["query"]
        n = min(int(arguments.get("n_results", 5)), 20)

        if name == "search_codebase":
            if not _code_collection:
                return [TextContent(type="text", text="Code index missing. Run: python scripts/index_code.py")]
            result = _search(_code_collection, query, n)

        elif name == "search_docs":
            if not _docs_collection:
                return [TextContent(type="text", text="Docs index missing. Run: python scripts/index_docs.py")]
            result = _search(_docs_collection, query, n)

        elif name == "search_tests":
            if not _tests_collection:
                return [TextContent(type="text", text="Tests index missing. Run: python scripts/index_tests.py")]
            result = _search(_tests_collection, query, n)

        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as exc:
        log.exception("Tool '%s' failed", name)
        return [TextContent(type="text", text=f"Error in {name}: {exc}")]


# ── Starlette / SSE wiring ────────────────────────────────────────────────────

sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    """SSE endpoint — matches IDG_LLM pattern for MCP 1.x + Starlette 1.x."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        return await server.run(streams[0], streams[1], server.create_initialization_options())


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAFoam Chroma RAG SSE MCP server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHROMA_MCP_PORT", settings.chroma_mcp_port)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    startup()
    log.info("Listening on http://%s:%d/sse", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
