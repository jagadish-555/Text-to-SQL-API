from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any



class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class TableDetail(BaseModel):
    relevance_score: float
    reason: str
    columns: List[str]


class RetrieveResponse(BaseModel):
    retrieved_tables: List[str]
    scores: List[float]
    confidence: float
    details: Dict[str, TableDetail]



class GenerateSQLRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    use_retrieved_context: bool = Field(default=True)


class ExecutionResult(BaseModel):
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    error: Optional[str]


class GenerateSQLResponse(BaseModel):
    sql: Optional[str]
    retrieved_tables: List[str]
    is_valid_syntax: bool
    parsing_errors: Optional[str]
    confidence: float
    prompt_used: str
    execution_result: Optional[ExecutionResult]



class BenchmarkMetrics(BaseModel):
    retrieval_recall_at_5: float
    retrieval_recall_at_10: float
    sql_exact_match_accuracy: float
    sql_execution_match_accuracy: float
    parsing_success_rate: float
    average_latency_ms: float


class ErrorAnalysis(BaseModel):
    retrieval_failures: int
    parsing_failures: int
    execution_failures: int
    logic_errors: int


class BenchmarkResponse(BaseModel):
    total_queries: int
    metrics: BenchmarkMetrics
    error_analysis: ErrorAnalysis
    