"""Index DAFoam test cases and OpenFOAM case config files into Chroma.

Indexes two categories from ../dafoam/tests/:
  1. Python test runner scripts (runRegTests_*.py, runUnitTests_*.py, testFuncs.py)
  2. OpenFOAM case config files: constant/, system/, 0/ directories
     (files without extension or with known OF extensions)

Test scripts are goldmines for "how do I configure solver X?" queries —
they contain the exact DAOPTION dictionaries used for each solver type.

Usage:
    python scripts/index_tests.py                  # use DAFOAM_REPO_PATH from .env
    python scripts/index_tests.py --repo ../dafoam
    python scripts/index_tests.py --rebuild
    python scripts/index_tests.py --cpu
"""
import argparse
import logging
import sys
from pathlib import Path

import chromadb
from git import InvalidGitRepositoryError, Repo
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings
from src.rag.embeddings import get_embed_model

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

PREFERRED_BRANCHES = ("main", "master")

# Python test scripts
PYTHON_EXTENSIONS = frozenset([".py"])

# OpenFOAM case directories (relative to any test case root)
OF_CASE_DIRS = {"constant", "system", "0"}

# OpenFOAM config files have no extension or specific ones
OF_EXTENSIONS = frozenset(["", ".cfg", ".foam", ".txt"])


def resolve_branch(repo: Repo) -> str:
    local_branches = {b.name for b in repo.branches}
    for name in PREFERRED_BRANCHES:
        if name in local_branches:
            return name
    return repo.active_branch.name


def is_openfoam_config(path: str) -> bool:
    """Return True if the file looks like an OpenFOAM dictionary."""
    parts = Path(path).parts
    # File must live inside constant/, system/, or 0/ directory
    return any(d in parts for d in OF_CASE_DIRS) and Path(path).suffix in OF_EXTENSIONS


def load_test_documents(repo_path: Path) -> list[Document]:
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        log.error("Not a git repo: %s", repo_path)
        return []

    branch = resolve_branch(repo)
    log.info("Indexing tests from: %s  (branch: %s)", repo_path.name, branch)

    try:
        commit = repo.commit(branch)
    except Exception as exc:
        log.error("Cannot resolve branch %s: %s", branch, exc)
        return []

    # Navigate to tests/ subtree
    try:
        tests_tree = commit.tree / "tests"
    except KeyError:
        log.error("No 'tests/' directory found in repo.")
        return []

    documents: list[Document] = []
    skipped = 0

    def _walk(tree, prefix: str) -> None:
        nonlocal skipped
        for item in tree:
            path = f"{prefix}/{item.name}"
            if item.type == "tree":
                _walk(item, path)
            elif item.type == "blob":
                ext = Path(item.name).suffix
                is_py = ext in PYTHON_EXTENSIONS
                is_of = is_openfoam_config(path)
                if not (is_py or is_of):
                    skipped += 1
                    continue
                try:
                    content = item.data_stream.read().decode("utf-8", errors="replace")
                except Exception:
                    skipped += 1
                    continue
                if not content.strip():
                    skipped += 1
                    continue

                doc_type = "python_test" if is_py else "openfoam_config"
                documents.append(Document(
                    text=content,
                    metadata={
                        "file_path": path,
                        "repo": repo_path.name,
                        "branch": branch,
                        "doc_type": doc_type,
                        "language": ext.lstrip(".") or "openfoam",
                    },
                ))

    _walk(tests_tree, "tests")
    log.info("  Loaded %d test documents (%d skipped)", len(documents), skipped)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DAFoam test cases into Chroma")
    parser.add_argument("--repo", default=None, help="Path to DAFoam repo (overrides .env)")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and re-index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    args = parser.parse_args()

    repo_path = Path(args.repo or settings.dafoam_repo_path).expanduser().resolve()
    if not repo_path.exists():
        log.error("Repo path does not exist: %s", repo_path)
        sys.exit(1)

    log.info("Loading embedding model...")
    embed_model = get_embed_model(force_cpu=args.cpu)

    db_cfg = Path(settings.chroma_tests_dir)
    db_path = str(db_cfg if db_cfg.is_absolute() else (PROJECT_ROOT / db_cfg).resolve())
    client = chromadb.PersistentClient(path=db_path)

    if args.rebuild:
        try:
            client.delete_collection(settings.chroma_collection_tests)
            log.info("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection_tests)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    documents = load_test_documents(repo_path)
    if not documents:
        log.error("No test documents loaded. Check DAFOAM_REPO_PATH.")
        sys.exit(1)

    splitter = SentenceSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    log.info("Chunking and embedding %d documents...", len(documents))
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    count = collection.count()
    log.info("Done. Collection '%s' has %d chunks.", settings.chroma_collection_tests, count)


if __name__ == "__main__":
    main()
