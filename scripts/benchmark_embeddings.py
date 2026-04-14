#!/usr/bin/env python3
"""Run a lightweight retrieval benchmark for local embedding models.

This script queries existing Chroma collections with a chosen embedding model
and reports hit-rate using keyword-based relevance checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings
from src.rag.embeddings import format_query_for_model, get_embed_model
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    collection: str
    query: str
    expected_keywords: tuple[str, ...]


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="code_daoption",
        collection="code",
        query="DAOPTION adjoint solver configuration in DAFoam",
        expected_keywords=("daoption", "adjoint"),
    ),
    BenchmarkCase(
        case_id="code_solver",
        collection="code",
        query="DASimpleFoam solver implementation and options",
        expected_keywords=("dasimplefoam", "solver"),
    ),
    BenchmarkCase(
        case_id="docs_install",
        collection="docs",
        query="DAFoam installation from source prerequisites",
        expected_keywords=("installation", "prerequisites"),
    ),
    BenchmarkCase(
        case_id="docs_openfoam",
        collection="docs",
        query="OpenFOAM AD build for DAFoam",
        expected_keywords=("openfoam", "ad"),
    ),
    BenchmarkCase(
        case_id="tests_options",
        collection="tests",
        query="test case DAOPTION setup for aerodynamic optimization",
        expected_keywords=("daoption", "optimization"),
    ),
    BenchmarkCase(
        case_id="tests_casefiles",
        collection="tests",
        query="OpenFOAM case configuration in tests system and constant folders",
        expected_keywords=("system", "constant"),
    ),
)


def get_collection(client: chromadb.PersistentClient, collection_key: str) -> chromadb.Collection:
    mapping = {
        "code": settings.chroma_collection_code,
        "docs": settings.chroma_collection_docs,
        "tests": settings.chroma_collection_tests,
    }
    return client.get_collection(mapping[collection_key])


def get_db_path(collection_key: str) -> str:
    mapping = {
        "code": settings.chroma_code_dir,
        "docs": settings.chroma_docs_dir,
        "tests": settings.chroma_tests_dir,
    }
    configured = Path(mapping[collection_key])
    return str(configured if configured.is_absolute() else (PROJECT_ROOT / configured).resolve())


def score_case(
    model,
    model_name: str,
    case: BenchmarkCase,
    top_k: int,
) -> dict:
    client = chromadb.PersistentClient(path=get_db_path(case.collection))
    collection = get_collection(client, case.collection)
    count = collection.count()
    if count == 0:
        return {
            "case_id": case.case_id,
            "collection": case.collection,
            "query": case.query,
            "status": "empty_collection",
            "hit": False,
            "top_distance": None,
            "matched_keyword": None,
        }

    query_text = format_query_for_model(case.query, model_name)
    query_embedding = model.get_text_embedding_batch([query_text])
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    lowered_keywords = tuple(k.lower() for k in case.expected_keywords)
    hit = False
    matched_keyword = None

    for doc, meta in zip(documents, metadatas):
        haystack = f"{doc}\n{json.dumps(meta)}".lower()
        for keyword in lowered_keywords:
            if keyword in haystack:
                hit = True
                matched_keyword = keyword
                break
        if hit:
            break

    top_distance = None
    if distances:
        top_distance = round(float(distances[0]), 4)

    return {
        "case_id": case.case_id,
        "collection": case.collection,
        "query": case.query,
        "status": "ok",
        "hit": hit,
        "top_distance": top_distance,
        "matched_keyword": matched_keyword,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local embedding models on existing Chroma indexes")
    parser.add_argument("--model", required=True, help="HF model id to test")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results per query")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    args = parser.parse_args()

    start = time.time()
    model = get_embed_model(model_name=args.model, force_cpu=args.cpu)

    rows: list[dict] = []
    for case in BENCHMARK_CASES:
        rows.append(score_case(model, args.model, case, top_k=args.top_k))

    attempted = sum(1 for r in rows if r["status"] == "ok")
    hits = sum(1 for r in rows if r["status"] == "ok" and r["hit"])
    hit_rate = (hits / attempted) if attempted else 0.0

    summary = {
        "model": args.model,
        "top_k": args.top_k,
        "attempted_cases": attempted,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "duration_sec": round(time.time() - start, 2),
        "results": rows,
    }

    print(json.dumps(summary, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
