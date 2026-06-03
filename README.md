# Text-to-SQL API

An enterprise-grade REST API that converts natural language questions into executable SQL queries using semantic retrieval and a large language model (LLM).

---

## System Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────────┐
│  FastAPI    │────▶│  Retrieval Pipeline (retrieval.py)       │
│  (main.py)  │     │  Stage 1: Bi-encoder coarse ranking      │
│             │     │           (all-MiniLM-L6-v2)             │
│             │     │  Stage 2: Cross-encoder reranking        │
│             │     │           (ms-marco-MiniLM-L-6-v2)       │
└─────────────┘     └──────────────────────────────────────────┘
     │                              │
     │                  Top-K relevant tables + schema
     │                              │
     ▼                              ▼
┌─────────────────────────────────────────────┐
│  LLM Pipeline (llm.py)                      │
│  - Build prompt with schema + few-shots     │
│  - Call Groq API (llama-3.3-70b-versatile)  │
│  - Clean & validate output                  │
│  - Hallucination check + auto-correction    │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  Validation (validation.py)                 │
│  - Parse with sqlglot                       │
│  - Enforce SELECT-only                      │
└─────────────────────────────────────────────┘
     │
     ▼
  SQL Response
```

---

## Components

### `main.py` — API Layer
The FastAPI application. Defines all endpoints, orchestrates the retrieval → generation → validation pipeline, and handles the benchmark runner.

### `retrieval.py` — Two-Stage Semantic Retrieval
Responsible for finding the most relevant database tables for a given question.

- **Stage 1 – Coarse Ranking (Bi-encoder):** At startup, all table schemas are embedded using `all-MiniLM-L6-v2` and L2-normalized. At query time, the question is embedded and scored against all tables via a fast dot-product (equivalent to cosine similarity on normalized vectors). The top 30 candidates are selected.

- **Stage 2 – Reranking (Cross-encoder):** The top 30 candidates are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`, which performs full attention between the question and each table description for a more accurate relevance score.

- Scores are min-max normalized to `[0, 1]` before being returned.

### `llm.py` — SQL Generation
Handles prompt construction and LLM interaction.

- **Prompt:** Includes the retrieved schema (table names + typed columns), SQLite-specific rules (e.g. `SQRT(AVG(x*x)-AVG(x)*AVG(x))` for `STDDEV`), and few-shot examples.
- **LLM:** Calls the Groq API. Primary model: `llama-3.3-70b-versatile`. Falls back to `llama-3.1-8b-instant` on rate-limit errors.
- **Self-Correction:**
  - Checks if the generated SQL is complete (balanced parentheses, no dangling clause).
  - Detects hallucinated column names using `sqlglot` AST parsing.
  - Retries with a correction hint if hallucinated columns are found.

### `validation.py` — SQL Validation
Parses the generated SQL using `sqlglot` and enforces:
- Non-empty SQL
- Valid parse tree
- Single statement only
- `SELECT`-only (no `INSERT`, `UPDATE`, `DELETE`, `DROP`)

### `execution.py` — SQL Execution
Executes the validated SQL against a local SQLite database (`db/{db_id}.db`). Returns up to 100 rows.

### `metrics.py` — Benchmark Metrics
Utility functions used by the `/benchmark` endpoint:
- `recall_at_k` — What fraction of gold tables appear in the top-K retrieved tables.
- `exact_match` — Normalized string match between predicted and gold SQL.
- `execution_match` — Row-level comparison of query results (order-insensitive).

### `models.py` — Pydantic Schemas
All request and response models for the API, including `RetrieveRequest`, `RetrieveResponse`, `GenerateSQLRequest`, `GenerateSQLResponse`, `BenchmarkResponse`, and sub-models.

---

## API Endpoints

### `GET /health`
Returns server status and the number of indexed tables.

**Response:**
```json
{
  "status": "ok",
  "tables_loaded": 381
}
```

---

### `POST /retrieve`
Finds the most relevant database tables for a natural language question.

**Request:**
```json
{
  "question": "Which departments have more than 100 enrolled students?",
  "top_k": 5
}
```

**Response:**
```json
{
  "retrieved_tables": ["departments", "enrollments"],
  "scores": [0.92, 0.87],
  "confidence": 0.90,
  "details": {
    "departments": {
      "relevance_score": 0.92,
      "reason": "Highly relevant based on context"
    },
    "enrollments": {
      "relevance_score": 0.87,
      "reason": "Likely needed for context or joining"
    }
  }
}
```

---

### `POST /generate-sql`
Generates a SQL query from a natural language question using retrieved schema context.

**Request:**
```json
{
  "question": "List all students enrolled in more than 3 courses",
  "use_retrieved_context": true
}
```

**Response:**
```json
{
  "sql": "SELECT d.dept_name, COUNT(e.student_id) AS total_students FROM departments d JOIN enrollments e ON d.dept_id = e.dept_id GROUP BY d.dept_name HAVING COUNT(e.student_id) > 100",
  "retrieved_tables": ["departments", "enrollments"],
  "is_valid_syntax": true,
  "parsing_errors": null,
  "confidence": 0.85,
  "prompt_used": "...full prompt sent to LLM..."
}
```

---

### `POST /benchmark`
Runs the full evaluation suite over `data/benchmark.json` (25 queries) and returns comprehensive metrics.

**Response:**
```json
{
  "total_queries": 25,
  "metrics": {
    "retrieval_recall_at_5": 0.88,
    "retrieval_recall_at_10": 0.92,
    "sql_exact_match_accuracy": 0.12,
    "sql_execution_match_accuracy": 0.44,
    "parsing_success_rate": 0.96,
    "average_latency_ms": 1823.4
  },
  "subtask_breakdown": {
    "multi_table_retrieval": 0.81,
    "column_mapping": 0.92,
    "join_detection": 0.75,
    "domain_knowledge": 0.40
  },
  "error_analysis": {
    "retrieval_failures": 0,
    "parsing_failures": 1,
    "execution_failures": 14,
    "logic_errors": 0
  }
}
```

---

## Running Locally

**1. Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set environment variables:**
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**3. Load the dataset:**
```bash
python data_loader.py
```
This will download the required datasets from Hugging Face and build the local SQLite databases and JSON metadata files.

**4. Start the server:**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

**Interactive docs:** `http://localhost:8000/docs`

---

## Data

| Path | Description |
|---|---|
| `data/schemas.json` | Table schema definitions (name, columns, types, db_id) |
| `data/benchmark.json` | 25 benchmark questions with gold SQL and gold tables |
| `db/{db_id}.db` | SQLite database files used for execution |

---

## Tech Stack

| Component | Library / Service |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Bi-encoder Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Cross-encoder Reranker | `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| SQL Parsing & Validation | `sqlglot` |
| Database | SQLite |
| Schema Validation | Pydantic v2 |
