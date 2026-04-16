# DAFoam RAG — MCP Server

> A local Chroma RAG MCP server for the [DAFoam](https://github.com/mdolab/dafoam) codebase.
> Indexes Python + C++ source, Sphinx documentation, test cases, and ecosystem tutorial repositories.
> Exposes four search tools over SSE for Copilot CLI, claude code, and so on.

---

## Quick Start

### 1. Prerequisites

- **Python 3.11** — required for the CUDA torch wheel (`cp311` only). Do NOT use 3.12/3.14.
- Network access to clone DAFoam into `data/repo_cache` (or set `DAFOAM_REPO_PATH` for a local override)

```powershell
# Install uv (fast Python version manager) if not already installed
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.11

# Create venv with Python 3.11
$py = "$env:APPDATA\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe"
& $py -m venv .venv
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Install GPU-accelerated PyTorch (CUDA 12.4)
# If pip can reach download.pytorch.org directly:
pip install -r requirements-cuda.txt

# Configure
copy .env.example .env
```

### 2. Build indexes

```powershell
# Index DAFoam Python + C++ source
python scripts\index_code.py

# Index documentation (clones DAFoam/DAFoam.github.io Sphinx source)
python scripts\index_docs.py

# Index test cases + OpenFOAM case configs
python scripts\index_tests.py

# Index tutorials + prerequisite ecosystem repositories (priority 1 by default)
python scripts\index_tutorials.py --max-priority 1
```

### 3. Start the MCP server

```powershell
.\scripts\Start-ChromaServer.ps1
```

Output:
```
dafoam-rag SSE server started (PID 12345).
  SSE URL:  http://127.0.0.1:29310/sse
```

### 4. Add to Copilot CLI

Merge this into `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "dafoam-rag": {
      "type": "sse",
      "url": "http://127.0.0.1:29310/sse"
    }
  }
}
```

See `config/mcp-config-entry.json` for the snippet.

### 5. Verify

```powershell
.\scripts\Start-ChromaServer.ps1 -Status
python scripts\test_e2e.py
```

---

## Architecture

```
Copilot CLI / Continue.dev
        │  SSE  http://127.0.0.1:29310/sse
        ▼
  chroma_sse_server.py   (MCP SSE server, port 29310)
        │
  ┌─────┼──────────┬────────────┐
  │     │          │            │
code  docs       tests     tutorials
Chroma × 4   (persistent, on-disk)
        │
  Snowflake/snowflake-arctic-embed-l-v2.0
  (local-first default with bge-m3 fallback)
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **MCP transport** | SSE (Starlette + uvicorn) | Persistent server, single model load |
| **Embeddings** | `snowflake-arctic-embed-l-v2.0` (+ `bge-m3` fallback) | Local retrieval embeddings |
| **Vector DBs** | Chroma (4 × persistent) | Code, docs, tests, tutorials |
| **MCP framework** | `mcp` Python SDK | Tool registration + SSE transport |

---

## MCP Tools

| Tool | Collection | Use when… |
|------|-----------|-----------|
| `search_codebase` | `dafoam_code` | Looking up DAFoam classes, methods, adjoint C++ |
| `search_docs` | `dafoam_docs` | Installation, tutorials, API reference, solver docs |
| `search_tests` | `dafoam_tests` | DAOPTION examples, solver case setups, reference configurations |
| `search_tutorials` | `dafoam_tutorials` | DAFoam tutorial repos and prerequisite ecosystem sources |

---

## Indexing

```powershell
# Source code (Python + C++)
python scripts\index_code.py                   # default: cached clone (mdolab/dafoam main)
python scripts\index_code.py --repo C:\path\to\dafoam  # custom path
python scripts\index_code.py --rebuild         # wipe and re-index
python scripts\index_code.py --cpu             # force CPU

# Documentation
python scripts\index_docs.py                   # clone DAFoam/DAFoam.github.io + index RST/MD
python scripts\index_docs.py --source scrape   # scrape dafoam.github.io HTML instead
python scripts\index_docs.py --rebuild

# Test cases + OpenFOAM configs
python scripts\index_tests.py
python scripts\index_tests.py --rebuild

# Ecosystem tutorials and prerequisite repositories
python scripts\index_tutorials.py --max-priority 1
python scripts\index_tutorials.py --max-priority 3 --include-prerequisites
python scripts\index_tutorials.py --rebuild
```

### Docs repo

Documentation is sourced from [`DAFoam/DAFoam.github.io`](https://github.com/DAFoam/DAFoam.github.io) —
the Sphinx source for https://dafoam.github.io/. The script clones it to `./data/docs_repo/`
and re-pulls on subsequent runs.

---

## Embedding Model

**Primary:** `Snowflake/snowflake-arctic-embed-l-v2.0`
- Local-first multilingual retrieval model
- 8 192-token context
- Balanced quality for code + docs queries

**Fallback:** `BAAI/bge-m3`

Control via `.env`:
```
EMBED_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0
EMBED_FALLBACK_MODEL=BAAI/bge-m3
EMBED_AUTO_DOWNLOAD_ON_MISS=true
```

---

## Server Management

```powershell
# Start
.\scripts\Start-ChromaServer.ps1

# Check status
.\scripts\Start-ChromaServer.ps1 -Status

# Stop
.\scripts\Start-ChromaServer.ps1 -Stop

# Custom port
.\scripts\Start-ChromaServer.ps1 -Port 29311

# View logs
Get-Content .\data\chroma_server_err.log -Tail 50
```

---

## Citation

If this project helps your research or workflow, please cite and acknowledge DAFoam:

- DAFoam website: https://dafoam.github.io/
- DAFoam repository: https://github.com/mdolab/dafoam

For official citation format, follow the guidance on the DAFoam project pages.

---

## License

This repository is licensed under **GNU GPL v3.0**.  
See `LICENSE` for details.

---

## Project Structure

```
dafoam-rag/
├── config/
│   ├── settings.py              ← Pydantic settings (.env)
│   ├── repository_sources.json  ← External repo sources for tutorials index
│   └── mcp-config-entry.json   ← Snippet for ~/.copilot/mcp-config.json
├── src/
│   ├── rag/
│   │   └── embeddings.py        ← CUDA-aware embedding factory
│   └── mcp/
│       └── chroma_sse_server.py ← SSE MCP server (port 29310)
├── scripts/
│   ├── index_code.py            ← Index DAFoam Python + C++
│   ├── index_docs.py            ← Index DAFoam docs (RST/MD or web scrape)
│   ├── index_tests.py           ← Index test cases + OF configs
│   ├── index_tutorials.py       ← Index ecosystem tutorial/prerequisite repos
│   ├── test_e2e.py              ← Smoke test
│   └── Start-ChromaServer.ps1  ← Start/stop/status
├── data/                        ← Chroma DBs + server logs (gitignored)
├── .env.example
├── requirements.txt
└── requirements-cuda.txt
```
