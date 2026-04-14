"""Embedding utilities for local model loading and query formatting."""
import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

log = logging.getLogger(__name__)


def format_query_for_model(query: str, model_name: str) -> str:
    """Format retrieval query text according to model-specific conventions."""
    name = model_name.lower()

    if "mxbai" in name:
        return f"Represent this sentence for searching relevant passages: {query}"
    if "e5" in name or "gte-" in name or "snowflake-arctic" in name:
        return f"query: {query}"

    # bge-m3 and jina-v3 generally work without explicit textual prefixes.
    return query


def get_embed_model(
    model_name: str | None = None,
    force_cpu: bool = False,
    embed_batch_size: int = 4,
) -> HuggingFaceEmbedding:
    """Return a HuggingFaceEmbedding using CUDA if available, else CPU.

    Tries the primary model first; if offline and unavailable, tries fallback.
    """
    from config.settings import settings

    primary = model_name or settings.embed_model
    fallback = settings.embed_fallback_model

    if not force_cpu and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("Using GPU for embeddings: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        device = "cpu"
        log.info("Using CPU for embeddings%s", " (forced)" if force_cpu else " (CUDA not available)")

    def _build(name: str) -> HuggingFaceEmbedding:
        # jina-embeddings-v2 uses a custom architecture that requires trust_remote_code
        needs_trust = "jina" in name.lower()
        kwargs = {"trust_remote_code": True} if needs_trust else {}
        return HuggingFaceEmbedding(
            model_name=name,
            device=device,
            embed_batch_size=embed_batch_size,
            model_kwargs=kwargs,
        )

    def _try_load(name: str) -> HuggingFaceEmbedding:
        """Try offline first; optionally allow one online download retry."""
        try:
            return _build(name)
        except Exception as exc:
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            if not (offline and settings.embed_auto_download_on_miss):
                raise

            log.warning(
                "Model '%s' not in local cache. Retrying with network enabled.", name
            )
            log.debug("Offline load failure: %s", exc)

            prev = os.environ.get("HF_HUB_OFFLINE")
            try:
                os.environ["HF_HUB_OFFLINE"] = "0"
                model = _build(name)
                log.info("Model '%s' downloaded and cached.", name)
                return model
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev

    # Try primary, then fallback
    try:
        model = _try_load(primary)
        log.info("Embedding model loaded: %s", primary)
        return model
    except Exception as primary_exc:
        if primary == fallback:
            raise
        log.warning(
            "Primary model '%s' failed (%s). Trying fallback: %s",
            primary, primary_exc, fallback,
        )
        model = _try_load(fallback)
        log.info("Fallback embedding model loaded: %s", fallback)
        return model
