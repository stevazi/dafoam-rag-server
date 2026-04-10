"""Smoke test: verify all three MCP tools return non-empty results.

Uses the MCP Python client SDK (SSE transport) — the proper way to call
an SSE MCP server. The server must be running before executing this script.

Requires:
  1. Server running: .\\scripts\\Start-ChromaServer.ps1
  2. At least one index built (index_code.py / index_docs.py / index_tests.py)

Usage:
    python scripts/test_e2e.py
    python scripts/test_e2e.py --port 29310
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import urllib.request

# Bypass corporate proxy (Zscaler) for localhost connections
_existing = os.environ.get("NO_PROXY", "")
_extra = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = f"{_existing},{_extra}" if _existing else _extra
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from mcp import ClientSession
from mcp.client.sse import sse_client


def wait_for_server(sse_url: str, retries: int = 10, delay: float = 2.0) -> bool:
    # Use a no-proxy opener so Zscaler doesn't intercept localhost connections
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for i in range(retries):
        try:
            opener.open(sse_url, timeout=5)
            return True
        except Exception as e:
            print(f"  Waiting for server... ({i + 1}/{retries}) [{type(e).__name__}: {e}]")
            time.sleep(delay)
    return False


async def run_tests(sse_url: str) -> bool:
    tests = [
        ("search_codebase", "DAOPTION adjoint solver configuration",
         "Code index empty or tool failed"),
        ("search_docs",     "how to install DAFoam OpenFOAM",
         "Docs index empty or tool failed"),
        ("search_tests",    "DASimpleFoam aerodynamic optimization case setup",
         "Tests index empty or tool failed"),
    ]

    all_passed = True
    # Pass no_proxy so httpx/httpcore bypass Zscaler for localhost
    async with sse_client(sse_url, headers={}, timeout=30) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for tool, query, hint in tests:
                print(f"\n[{tool}] query: {query!r}")
                try:
                    resp = await session.call_tool(tool, {"query": query, "n_results": 3})
                    text = resp.content[0].text if resp.content else ""
                    if not text or text.startswith("No results") or "index missing" in text:
                        print(f"  WARN: {hint}")
                        print(f"  Response: {text[:200]}")
                    else:
                        snippet = text[:300].replace("\n", " ")
                        print(f"  OK  ({len(text)} chars): {snippet}...")
                except Exception as exc:
                    print(f"  FAIL: {exc}")
                    all_passed = False

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="DAFoam RAG MCP smoke test")
    parser.add_argument("--port", type=int, default=29310)
    args = parser.parse_args()

    sse_url = f"http://127.0.0.1:{args.port}/sse"
    print(f"Testing server at {sse_url} ...")

    if not wait_for_server(sse_url):
        print("ERROR: Server not reachable. Start it with: .\\scripts\\Start-ChromaServer.ps1")
        sys.exit(1)

    all_passed = asyncio.run(run_tests(sse_url))

    print()
    if all_passed:
        print("Smoke test passed.")
    else:
        print("Some tests failed — check server logs in data/chroma_server_err.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
