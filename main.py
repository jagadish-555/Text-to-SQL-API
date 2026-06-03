import json
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    RetrieveRequest, RetrieveResponse, TableDetail,
    GenerateSQLRequest, GenerateSQLResponse, ExecutionResult,
    BenchmarkResponse, BenchmarkMetrics, SubtaskBreakdown, ErrorAnalysis,
)
from retrieval import load_and_embed_schemas, retrieve_tables
import retrieval
from llm import generate_sql
from validation import validate_sql
from execution import execute_sql
from metrics import recall_at_k, exact_match, execution_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading and embedding table schemas...")
    load_and_embed_schemas("data/schemas.json")
    logger.info(f"Ready — {len(retrieval.TABLE_EMBEDDINGS)} tables indexed")
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="Enterprise Text-to-SQL API",
    description="Natural language → SQL using semantic retrieval and an LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tables_loaded": len(retrieval.TABLE_EMBEDDINGS) if retrieval.TABLE_EMBEDDINGS is not None else 0,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest):
    start = time.time()
    logger.info(f"/retrieve question={request.question!r} top_k={request.top_k}")

    results = retrieve_tables(request.question, top_k=request.top_k)

    if not results:
        raise HTTPException(status_code=500, detail="Retrieval returned no results")

    avg_confidence = sum(r["score"] for r in results) / len(results)
    latency_ms = round((time.time() - start) * 1000, 1)

    logger.info(f"/retrieve done tables={len(results)} latency={latency_ms}ms")

    return RetrieveResponse(
        retrieved_tables=[r["table_name"] for r in results],
        scores=[r["score"] for r in results],
        confidence=round(avg_confidence, 3),
        details={
            r["table_name"]: TableDetail(
                relevance_score=r["score"],
                reason=r["reason"],
            )
            for r in results
        },
    )


@app.post("/generate-sql", response_model=GenerateSQLResponse)
def generate_sql_endpoint(request: GenerateSQLRequest):
    start = time.time()
    logger.info(f"/generate-sql question={request.question!r}")

    if request.use_retrieved_context:
        retrieved = retrieve_tables(request.question, top_k=5)
    else:
        retrieved = []
        logger.warning("Generating SQL without schema context — quality will be lower")

    llm_result = generate_sql(request.question, retrieved)

    if llm_result["error"] and not llm_result["sql"]:
        raise HTTPException(
            status_code=500,
            detail=f"LLM failed: {llm_result['error']}",
        )

    validation = validate_sql(llm_result["sql"] or "")



    avg_confidence = (
        sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0.0
    )
    latency_ms = round((time.time() - start) * 1000, 1)

    logger.info(f"/generate-sql done valid={validation['is_valid']} latency={latency_ms}ms")

    return GenerateSQLResponse(
        sql=llm_result["sql"],
        retrieved_tables=[r["table_name"] for r in retrieved],
        is_valid_syntax=validation["is_valid"],
        parsing_errors=validation["errors"],
        confidence=round(avg_confidence, 3),
        prompt_used=llm_result["prompt_used"],
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark_endpoint():
    import re as _re
    logger.info("Benchmark started")

    with open("data/benchmark.json", "r") as f:
        questions = json.load(f)

    recall_5, recall_10 = [], []
    exact_scores, exec_scores, parse_scores = [], [], []
    latencies = []

    retrieval_failures = 0
    parsing_failures = 0
    execution_failures = 0
    logic_errors = 0

    multi_table_recall5 = []
    column_map_scores = []
    join_detection_scores = []
    domain_knowledge_scores = []

    ADVANCED_PATTERN = _re.compile(
        r'\bSTDDEV\b|\bEXP\b|\bLN\b|\bHAVING\b|\bWITH\b|\bSTDDEV_POP\b',
        _re.IGNORECASE
    )

    for i, item in enumerate(questions):
        logger.info(f"Benchmark {i+1}/{len(questions)}: {item['question'][:60]}")
        t0 = time.time()

        question = item["question"]
        gold_sql = item["gold_sql"]
        gold_tables = item.get("gold_tables", [])
        db_id = item.get("db_id")
        is_multi_table = len(gold_tables) > 1
        is_advanced = bool(ADVANCED_PATTERN.search(gold_sql))

        retrieved = retrieve_tables(question, top_k=10)
        if not retrieved:
            retrieval_failures += 1

        r5 = recall_at_k(retrieved, gold_tables, 5)
        recall_5.append(r5)
        recall_10.append(recall_at_k(retrieved, gold_tables, 10))

        if is_multi_table:
            multi_table_recall5.append(r5)

        llm_result = generate_sql(question, retrieved[:5])
        gen_sql = llm_result.get("sql") or ""

        val = validate_sql(gen_sql)
        is_valid = val["is_valid"]
        parse_scores.append(1.0 if is_valid else 0.0)
        if not is_valid:
            parsing_failures += 1

        exact_scores.append(1.0 if exact_match(gen_sql, gold_sql) else 0.0)

        exec_match_val = 0.0
        if is_valid and gen_sql and db_id:
            pred = execute_sql(gen_sql, db_id=db_id)
            gold = execute_sql(gold_sql, db_id=db_id)

            if not pred["success"]:
                execution_failures += 1
            else:
                matched = execution_match(pred["rows"], gold["rows"])
                exec_match_val = 1.0 if matched else 0.0
                if not matched:
                    logic_errors += 1

            exec_scores.append(exec_match_val)
        else:
            exec_scores.append(0.0)

        if is_valid and gen_sql:
            retrieved_names = {r["table_name"].upper() for r in retrieved[:5]}
            sql_upper = gen_sql.upper()
            gen_tables_used = {
                r["table_name"].upper()
                for r in retrieved[:5]
                if r["table_name"].upper() in sql_upper
            }
            column_map_scores.append(1.0 if gen_tables_used else 0.0)
        else:
            column_map_scores.append(0.0)

        if is_multi_table:
            has_join = bool(_re.search(r'\bJOIN\b', gen_sql, _re.IGNORECASE))
            join_detection_scores.append(1.0 if has_join else 0.0)

        if is_advanced:
            domain_knowledge_scores.append(exec_match_val)

        latencies.append((time.time() - t0) * 1000)

    def avg(lst):
        return round(sum(lst) / max(len(lst), 1), 3)

    logger.info(
        f"Benchmark done recall@5={avg(recall_5)} "
        f"exact={avg(exact_scores)} exec={avg(exec_scores)}"
    )

    return BenchmarkResponse(
        total_queries=len(questions),
        metrics=BenchmarkMetrics(
            retrieval_recall_at_5=avg(recall_5),
            retrieval_recall_at_10=avg(recall_10),
            sql_exact_match_accuracy=avg(exact_scores),
            sql_execution_match_accuracy=avg(exec_scores),
            parsing_success_rate=avg(parse_scores),
            average_latency_ms=avg(latencies),
        ),
        subtask_breakdown=SubtaskBreakdown(
            multi_table_retrieval=avg(multi_table_recall5),
            column_mapping=avg(column_map_scores),
            join_detection=avg(join_detection_scores),
            domain_knowledge=avg(domain_knowledge_scores),
        ),
        error_analysis=ErrorAnalysis(
            retrieval_failures=retrieval_failures,
            parsing_failures=parsing_failures,
            execution_failures=execution_failures,
            logic_errors=logic_errors,
        ),
    )