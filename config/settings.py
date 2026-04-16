"""Application settings loaded from .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # DAFoam source repository settings
    dafoam_repo_url: str = "https://github.com/mdolab/dafoam.git"
    dafoam_repo_branch: str = "main"
    dafoam_repo_path: str = ""  # optional local override; if empty, use cached clone

    # Chroma DB directories
    chroma_code_dir: str = "./data/chroma_code"
    chroma_docs_dir: str = "./data/chroma_docs"
    chroma_tests_dir: str = "./data/chroma_tests"
    chroma_tutorials_dir: str = "./data/chroma_tutorials"
    repo_cache_dir: str = "./data/repo_cache"

    # Chroma collection names
    chroma_collection_code: str = "dafoam_code"
    chroma_collection_docs: str = "dafoam_docs"
    chroma_collection_tests: str = "dafoam_tests"
    chroma_collection_tutorials: str = "dafoam_tutorials"

    # Embeddings
    embed_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    embed_fallback_model: str = "BAAI/bge-m3"
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
