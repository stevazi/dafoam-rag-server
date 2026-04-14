#!/usr/bin/env python3
"""Compare local embedding models on a small retrieval discrimination benchmark."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.embeddings import format_query_for_model, get_embed_model


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    positive: str
    negatives: tuple[str, ...]


CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        case_id="daoption_config",
        query="How do I configure DAOPTION for aerodynamic optimization?",
        positive="DAOPTION dictionary controls optimizer, design variables, and objective settings in DAFoam cases.",
        negatives=("Sourdough bread fermentation and hydration ratio guide.", "Italian pasta cooking times and sauce pairing."),
    ),
    RetrievalCase(
        case_id="openfoam_ad",
        query="How to build OpenFOAM AD variants for DAFoam?",
        positive="DAFoam installation guide includes OpenFOAM-v1812, ADR, and ADF build workflows.",
        negatives=("Kubernetes rolling updates and blue-green deployment strategy.", "Photography exposure triangle basics."),
    ),
    RetrievalCase(
        case_id="petsc_setup",
        query="Which PETSc version is used in DAFoam source installation?",
        positive="DAFoam source installation uses PETSc 3.15.5 with petsc4py integration.",
        negatives=("Node.js package manager lockfile behavior.", "How to prune houseplants in spring."),
    ),
    RetrievalCase(
        case_id="solver_query",
        query="Where can I find DASimpleFoam test setup examples?",
        positive="DAFoam regression tests contain DASimpleFoam cases and DAOPTION example dictionaries.",
        negatives=("Neural style transfer overview with PyTorch.", "Road cycling cadence and gear ratio advice."),
    ),
    RetrievalCase(
        case_id="hisa_dependency",
        query="Is Hisa integrated as a dependency in DAFoam?",
        positive="Hisa4DAFoam is built as an optional high-speed CFD dependency in DAFoam setup.",
        negatives=("MySQL indexing and transaction isolation tutorial.", "Guitar barre chord finger exercises."),
    ),
    RetrievalCase(
        case_id="tutorial_repo",
        query="Which repository has DAFoam practical examples?",
        positive="DAFoam tutorials repository provides practical optimization examples and case files.",
        negatives=("Convolutional neural network architecture notes.", "Travel packing checklist for winter trips."),
    ),
)

DEFAULT_MODELS = (
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v3",
    "mixedbread-ai/mxbai-embed-large-v1",
    "intfloat/multilingual-e5-large",
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def evaluate_model(model_name: str, force_cpu: bool) -> dict:
    started = time.time()
    model = get_embed_model(model_name=model_name, force_cpu=force_cpu)

    case_rows: list[dict] = []
    for case in CASES:
        query_text = format_query_for_model(case.query, model_name)
        q_vec = model.get_text_embedding(query_text)
        p_vec = model.get_text_embedding(case.positive)
        n_vecs = [model.get_text_embedding(n) for n in case.negatives]

        pos_sim = cosine(q_vec, p_vec)
        neg_sims = [cosine(q_vec, n) for n in n_vecs]
        max_neg = max(neg_sims)
        hit = pos_sim > max_neg

        case_rows.append(
            {
                "case_id": case.case_id,
                "hit": hit,
                "pos_sim": round(pos_sim, 4),
                "max_neg_sim": round(max_neg, 4),
                "margin": round(pos_sim - max_neg, 4),
            }
        )

    hits = sum(1 for r in case_rows if r["hit"])
    avg_margin = sum(r["margin"] for r in case_rows) / len(case_rows)

    return {
        "model": model_name,
        "cases": len(case_rows),
        "hits": hits,
        "accuracy": round(hits / len(case_rows), 4),
        "avg_margin": round(avg_margin, 4),
        "duration_sec": round(time.time() - started, 2),
        "details": case_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local embedding model bake-off")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS), help="HF model ids to compare")
    parser.add_argument("--out", default="data/local_embedding_bakeoff.json", help="Output JSON path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    results: list[dict] = []
    for model_name in args.models:
        try:
            result = evaluate_model(model_name, force_cpu=args.cpu)
            results.append(result)
            print(f"[OK] {model_name} accuracy={result['accuracy']} avg_margin={result['avg_margin']}")
        except Exception as exc:
            print(f"[FAIL] {model_name}: {exc}")
            results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    ranked = [r for r in results if r.get("status") != "failed"]
    ranked.sort(key=lambda r: (r["accuracy"], r["avg_margin"]), reverse=True)

    payload = {
        "benchmark": "local_discrimination_v1",
        "timestamp_epoch": int(time.time()),
        "models_tested": args.models,
        "ranked": ranked,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
