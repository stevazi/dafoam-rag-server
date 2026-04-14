#!/usr/bin/env python3
"""Index additional DAFoam ecosystem repositories into a tutorials collection."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import chromadb
from git import Repo
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings
from src.rag.embeddings import get_embed_model

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

REPO_CONFIG_PATH = PROJECT_ROOT / "config" / "repository_sources.json"
REPO_CACHE_DIR = PROJECT_ROOT / "data" / "repo_cache"
ALLOWED_EXTENSIONS = frozenset(
    [
        ".py",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".C",
        ".H",
        ".md",
        ".rst",
        ".txt",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".conf",
        ".sh",
        ".mk",
        ".cmake",
    ]
)
SKIP_DIRS = {".git", ".github", ".idea", ".vscode", "build", "dist", "__pycache__", ".pytest_cache"}


def load_repo_sources() -> list[dict]:
    data = json.loads(REPO_CONFIG_PATH.read_text(encoding="utf-8"))
    return data.get("sources", [])


def sync_repo(repo_entry: dict) -> Path:
    repo_name = repo_entry["name"]
    url = repo_entry["url"]
    branch = repo_entry.get("branch", "main")
    repo_dir = REPO_CACHE_DIR / repo_name.replace("/", "__")

    if repo_dir.exists():
        try:
            repo = Repo(repo_dir)
            repo.git.fetch("--all", "--prune")
            repo.git.checkout(branch)
            repo.remotes.origin.pull()
            log.info("Updated %s (%s)", repo_name, branch)
        except Exception as exc:
            log.warning("Failed to update %s (%s). Re-cloning.", repo_name, exc)
            shutil.rmtree(repo_dir, ignore_errors=True)

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(url, repo_dir, branch=branch, depth=1)
        log.info("Cloned %s (%s)", repo_name, branch)

    return repo_dir


def build_documents(repo_entry: dict, repo_dir: Path) -> list[Document]:
    repo_name = repo_entry["name"]
    category = repo_entry.get("category", "tutorials")
    priority = repo_entry.get("priority", 99)
    docs: list[Document] = []

    for file_path in repo_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        if file_path.suffix not in ALLOWED_EXTENSIONS:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue

        rel = file_path.relative_to(repo_dir)
        docs.append(
            Document(
                text=text,
                metadata={
                    "repo_name": repo_name,
                    "category": category,
                    "priority": priority,
                    "file_path": str(rel),
                    "source": "repo_sources",
                },
            )
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DAFoam ecosystem repositories")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate tutorials collection")
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    parser.add_argument("--max-priority", type=int, default=2, help="Include repositories with priority <= N")
    parser.add_argument(
        "--include-prerequisites",
        action="store_true",
        help="Include category=prerequisite repositories (large set).",
    )
    args = parser.parse_args()

    sources = load_repo_sources()
    selected = []
    for src in sources:
        if src.get("priority", 99) > args.max_priority:
            continue
        if src.get("category") == "prerequisite" and not args.include_prerequisites:
            continue
        selected.append(src)

    if not selected:
        log.error("No repositories selected. Adjust --max-priority/--include-prerequisites.")
        sys.exit(1)

    embed_model = get_embed_model(force_cpu=args.cpu)

    db_cfg = Path(settings.chroma_tutorials_dir)
    db_path = str(db_cfg if db_cfg.is_absolute() else (PROJECT_ROOT / db_cfg).resolve())
    client = chromadb.PersistentClient(path=db_path)
    if args.rebuild:
        try:
            client.delete_collection(settings.chroma_collection_tutorials)
            log.info("Deleted existing tutorials collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection_tutorials)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap)

    all_docs: list[Document] = []
    for src in selected:
        repo_dir = sync_repo(src)
        docs = build_documents(src, repo_dir)
        log.info("Prepared %d docs from %s", len(docs), src["name"])
        all_docs.extend(docs)

    if not all_docs:
        log.error("No documents found in selected repositories.")
        sys.exit(1)

    log.info("Chunking and embedding %d documents...", len(all_docs))
    VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_ctx,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    log.info("Done. Collection '%s' has %d chunks.", settings.chroma_collection_tutorials, collection.count())


if __name__ == "__main__":
    main()
