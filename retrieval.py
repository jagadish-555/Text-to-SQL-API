import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

TABLE_DATA = []
TABLE_EMBEDDINGS = None


def _build_rich_description(table: dict) -> str:
    table_name = table["table_name"]
    db_id = table.get("db_id", "")
    readable_name = table_name.replace("_", " ")

    col_details = table.get("column_details", [])
    if col_details:
        col_parts = [
            f"{cd['name'].replace('_', ' ')} ({cd['type']})"
            for cd in col_details
        ]
    else:
        col_parts = [c.replace("_", " ") for c in table.get("columns", [])]

    col_str = ", ".join(col_parts)
    raw_cols = " ".join(c.replace("_", " ") for c in table.get("columns", []))

    description = (
        f"Table {readable_name} from the {db_id} database. "
        f"Columns: {col_str}. "
        f"Keywords: {raw_cols}."
    )
    return description


def load_and_embed_schemas(path="data/schemas.json"):
    global TABLE_DATA, TABLE_EMBEDDINGS

    with open(path, "r") as f:
        schemas = json.load(f)

    TABLE_DATA = list(schemas.values())
    descriptions = [_build_rich_description(t) for t in TABLE_DATA]
    print(f"Embedding {len(descriptions)} table descriptions for Stage 1...")
    TABLE_EMBEDDINGS = embedding_model.encode(descriptions, show_progress_bar=True)
    print("Embeddings ready.")


def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def retrieve_tables(question: str, top_k: int = 5, candidate_pool_size: int = 30) -> list:
    if TABLE_EMBEDDINGS is None:
        raise RuntimeError("Schemas not loaded. Call load_and_embed_schemas() first.")

    question_vec = embedding_model.encode(question)

    coarse_scores = [
        cosine_similarity(question_vec, TABLE_EMBEDDINGS[i])
        for i in range(len(TABLE_DATA))
    ]

    candidate_indices = sorted(
        range(len(coarse_scores)),
        key=lambda i: coarse_scores[i],
        reverse=True
    )[:candidate_pool_size]

    rerank_pairs = [
        [question, _build_rich_description(TABLE_DATA[idx])]
        for idx in candidate_indices
    ]

    rerank_scores = reranker_model.predict(rerank_pairs)

    reranked_results = list(zip(candidate_indices, rerank_scores))
    reranked_results.sort(key=lambda x: x[1], reverse=True)

    final_top = reranked_results[:top_k]

    raw_scores = [s for _, s in reranked_results[:top_k]]
    score_min = min(raw_scores) if raw_scores else 0.0
    score_max = max(raw_scores) if raw_scores else 1.0
    score_range = score_max - score_min if score_max != score_min else 1.0

    results = []
    for idx, score in final_top:
        table = TABLE_DATA[idx]
        normalized = round(float((score - score_min) / score_range), 4)
        reason = "Highly relevant based on context" if normalized >= 0.75 else "Likely needed for context or joining"
        results.append({
            "table_name": table["table_name"],
            "db_id": table["db_id"],
            "columns": table["columns"],
            "column_details": table.get("column_details", []),
            "score": normalized,
            "reason": reason,
        })

    return results