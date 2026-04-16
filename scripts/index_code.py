"""Index DAFoam Python and C++ source into Chroma.

By default, clones/pulls DAFoam into local cache and reads from there via
gitpython (working tree is never modified). Chunks and stores in chroma_code DB.

Usage:
    python scripts/index_code.py                   # use cached clone from REPO_CACHE_DIR
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
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# File extensions to index
CODE_EXTENSIONS = frozenset([
    # Python
    ".py",
    # C++ / OpenFOAM (capital .C and .H are OpenFOAM convention)
    ".C", ".H", ".cpp", ".h", ".c",
    # Config / build
    ".yml", ".yaml", ".cfg", ".conf",
])

PREFERRED_BRANCHES = ("main", "master")


def resolve_cache_repo_path() -> Path:
    cache_cfg = Path(settings.repo_cache_dir)
    cache_root = cache_cfg if cache_cfg.is_absolute() else (PROJECT_ROOT / cache_cfg).resolve()
    return cache_root / "mdolab__dafoam"


def sync_cached_repo() -> Path:
    repo_dir = resolve_cache_repo_path()
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists():
        repo = Repo(repo_dir)
        repo.remotes.origin.fetch()
        target = settings.dafoam_repo_branch
        if f"origin/{target}" in [ref.name for ref in repo.refs]:
            repo.git.checkout(target)
            repo.git.pull("origin", target)
        else:
            repo.remotes.origin.pull()
        log.info("Updated cached DAFoam repo: %s", repo_dir)
        return repo_dir

    Repo.clone_from(
        settings.dafoam_repo_url,
        repo_dir,
        branch=settings.dafoam_repo_branch,
        depth=1,
    )
    log.info("Cloned DAFoam repo to cache: %s", repo_dir)
    return repo_dir


def resolve_repo_path(repo_override: str | None) -> Path:
    if repo_override:
        return Path(repo_override).expanduser().resolve()
    if settings.dafoam_repo_path.strip():
        return Path(settings.dafoam_repo_path).expanduser().resolve()
    return sync_cached_repo()


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
    parser.add_argument("--repo", default=None, help="Path to DAFoam repo (overrides cache and .env)")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and re-index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    args = parser.parse_args()

    repo_path = resolve_repo_path(args.repo)
    if not repo_path.exists():
        log.error("Repo path does not exist: %s", repo_path)
        sys.exit(1)

    log.info("Loading embedding model...")
    embed_model = get_embed_model(force_cpu=args.cpu)

    db_cfg = Path(settings.chroma_code_dir)
    db_path = str(db_cfg if db_cfg.is_absolute() else (PROJECT_ROOT / db_cfg).resolve())
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
