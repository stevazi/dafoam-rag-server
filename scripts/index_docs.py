"""Index DAFoam documentation into Chroma.

Two source modes:
  --source repo (default)
      Clone/pull github.com/DAFoam/DAFoam.github.io into ./data/docs_repo/
      and index all .rst and .md files.

  --source scrape
      Web-scrape https://dafoam.github.io/ and index page text.
      Use when network access is limited or a local clone is preferred.

Usage:
    python scripts/index_docs.py                        # repo mode (default)
    python scripts/index_docs.py --source scrape        # web scrape
    python scripts/index_docs.py --rebuild              # wipe and re-index
    python scripts/index_docs.py --cpu                  # force CPU embeddings
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings
from src.rag.embeddings import get_embed_model

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DOCS_REPO_URL = "https://github.com/DAFoam/DAFoam.github.io.git"
DOCS_REPO_DIR = PROJECT_ROOT / "data" / "docs_repo"
DOCS_BASE_URL = "https://dafoam.github.io/"
DOC_EXTENSIONS = frozenset([".rst", ".md"])

# Paths inside the docs repo to skip (build artefacts, assets)
SKIP_DIRS = {"_build", "_static", "_images", ".git"}


def load_from_repo() -> list[Document]:
    """Clone or pull DAFoam/DAFoam.github.io and index RST + MD files."""
    try:
        from git import Repo, InvalidGitRepositoryError
    except ImportError:
        log.error("gitpython not installed. Run: pip install gitpython")
        sys.exit(1)

    local_dir = DOCS_REPO_DIR.resolve()

    if local_dir.exists():
        log.info("Pulling latest docs from %s ...", DOCS_REPO_URL)
        try:
            repo = Repo(local_dir)
            repo.remotes.origin.pull()
        except Exception as exc:
            log.warning("Pull failed (%s) — using existing clone.", exc)
    else:
        log.info("Cloning %s → %s ...", DOCS_REPO_URL, local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(DOCS_REPO_URL, local_dir, depth=1)

    documents: list[Document] = []
    for f in local_dir.rglob("*"):
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        if f.suffix not in DOC_EXTENSIONS:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not content:
            continue
        rel = f.relative_to(local_dir)
        documents.append(Document(
            text=content,
            metadata={
                "file_path": str(rel),
                "source": "DAFoam.github.io",
                "format": f.suffix.lstrip("."),
            },
        ))

    log.info("Loaded %d documents from docs repo.", len(documents))
    return documents


def load_from_scrape() -> list[Document]:
    """Scrape HTML pages from dafoam.github.io and return as Documents."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("requests/beautifulsoup4 not installed. Run: pip install requests beautifulsoup4")
        sys.exit(1)

    visited: set[str] = set()
    queue: list[str] = [DOCS_BASE_URL]
    documents: list[Document] = []

    log.info("Scraping %s ...", DOCS_BASE_URL)

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Skipping %s: %s", url, exc)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract main text content
        main = soup.find("div", {"role": "main"}) or soup.find("article") or soup.body
        if not main:
            continue
        text = main.get_text(separator="\n", strip=True)
        if len(text) < 100:
            continue

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        documents.append(Document(
            text=text,
            metadata={
                "url": url,
                "title": title,
                "source": "dafoam.github.io",
            },
        ))
        log.debug("Scraped: %s (%d chars)", url, len(text))

        # Discover more pages under the same domain
        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            if href.startswith("/") and not href.startswith("//"):
                full = DOCS_BASE_URL.rstrip("/") + href
            elif href.startswith(DOCS_BASE_URL):
                full = href
            else:
                continue
            # Drop anchors and query strings
            full = full.split("#")[0].split("?")[0]
            if full not in visited and full not in queue:
                queue.append(full)

        time.sleep(0.2)  # polite crawling delay

    log.info("Scraped %d pages.", len(documents))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DAFoam docs into Chroma")
    parser.add_argument("--source", choices=["repo", "scrape"], default="repo",
                        help="repo: clone DAFoam/DAFoam.github.io; scrape: crawl dafoam.github.io")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and re-index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    args = parser.parse_args()

    log.info("Loading embedding model...")
    embed_model = get_embed_model(force_cpu=args.cpu)

    db_cfg = Path(settings.chroma_docs_dir)
    db_path = str(db_cfg if db_cfg.is_absolute() else (PROJECT_ROOT / db_cfg).resolve())
    client = chromadb.PersistentClient(path=db_path)

    if args.rebuild:
        try:
            client.delete_collection(settings.chroma_collection_docs)
            log.info("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection_docs)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    if args.source == "scrape":
        documents = load_from_scrape()
    else:
        documents = load_from_repo()

    if not documents:
        log.error("No documents loaded.")
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
    log.info("Done. Collection '%s' has %d chunks.", settings.chroma_collection_docs, count)


if __name__ == "__main__":
    main()
