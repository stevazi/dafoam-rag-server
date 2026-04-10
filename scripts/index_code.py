"""Index DAFoam Python and C++ source into Chroma.

Reads from the main/master branch of the local DAFoam clone via gitpython
(working tree is never modified). Chunks and stores in chroma_code DB.

Usage:
    python scripts/index_code.py                   # use DAFOAM_REPO_PATH from .env
    python scripts/index_code.py --repo ../dafoam  # override repo path
    python scripts/index_code.py --rebuild         # wipe and re-index
    python scripts/index_code.py --cpu             # force CPU embeddings
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

# File extensions to index
CODE_EXTENSIONS = frozenset([
    # Python
    ".py",
    # C++ / OpenFOAM (capital .C and .H are OpenFOAM convention)
    ".C", ".H", ".cpp", ".h", ".c",
    # Config / build
    ".yml", ".yaml", ".cfg", ".conf",
    # Documentation in repo
    ".md", ".rst",
])

PREFERRED_BRANCHES = ("main", "master")


def resolve_branch(repo: Repo) -> str:
    local_branches = {b.name for b in repo.branches}
    for name in PREFERRED_BRANCHES:
        if name in local_branches:
            return name
    fallback = repo.active_branch.name
    log.warning("No main/master branch — using active branch: %s", fallback)
    return fallback


def load_repo_documents(repo_path: Path) -> list[Document]:
    """Read all matching files from the default branch via git object store."""
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        log.error("Not a git repo: %s", repo_path)
        return []

    branch = resolve_branch(repo)
    log.info("Indexing repo: %s  (branch: %s)", repo_path.name, branch)

    try:
        commit = repo.commit(branch)
    except Exception as exc:
        log.error("Cannot resolve branch %s: %s", branch, exc)
        return []

    documents: list[Document] = []
    skipped = 0

    def _walk(tree, prefix: str = "") -> None:
        nonlocal skipped
        for item in tree:
            path = f"{prefix}/{item.name}" if prefix else item.name
            if item.type == "tree":
                _walk(item, path)
            elif item.type == "blob":
                if Path(item.name).suffix not in CODE_EXTENSIONS:
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
                documents.append(Document(
                    text=content,
                    metadata={
                        "file_path": path,
                        "repo": repo_path.name,
                        "branch": branch,
                        "language": Path(item.name).suffix.lstrip("."),
                    },
                ))

    _walk(commit.tree)
    log.info("  Loaded %d documents (%d skipped)", len(documents), skipped)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DAFoam source into Chroma")
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

    db_path = str(Path(settings.chroma_code_dir))
    client = chromadb.PersistentClient(path=db_path)

    if args.rebuild:
        try:
            client.delete_collection(settings.chroma_collection_code)
            log.info("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection_code)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    documents = load_repo_documents(repo_path)
    if not documents:
        log.error("No documents loaded. Check DAFOAM_REPO_PATH.")
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
    log.info("Done. Collection '%s' has %d chunks.", settings.chroma_collection_code, count)


if __name__ == "__main__":
    main()
