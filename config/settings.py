"""Application settings loaded from .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # DAFoam source repository to index
    dafoam_repo_path: str = "../dafoam"

    # Chroma DB directories
    chroma_code_dir: str = "./data/chroma_code"
    chroma_docs_dir: str = "./data/chroma_docs"
    chroma_tests_dir: str = "./data/chroma_tests"

    # Chroma collection names
    chroma_collection_code: str = "dafoam_code"
    chroma_collection_docs: str = "dafoam_docs"
    chroma_collection_tests: str = "dafoam_tests"

    # Embeddings — e5-large is primary (cached, proven working, transformers-compatible)
    # Jina dropped as primary: persistent 'BertConfig.attn_implementation' compat issues
    embed_model: str = "intfloat/multilingual-e5-large"
    embed_fallback_model: str = "jinaai/jina-embeddings-v2-base-code"
    embed_auto_download_on_miss: bool = True

    # MCP server port (29310 avoids conflict with IDG_LLM at 29300)
    chroma_mcp_port: int = 29310

    # RAG retrieval parameters
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5

    # Logging
    log_level: str = "INFO"


settings = Settings()
