def recall_at_k(retrieved: list, gold_tables: list, k: int) -> float:
    if not gold_tables:
        return 0.0

    retrieved_names = {r["table_name"].lower() for r in retrieved[:k]}
    gold_names = {t.lower() for t in gold_tables}

    found = len(gold_names & retrieved_names)
    return found / len(gold_names)


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    if not predicted_sql or not gold_sql:
        return False

    def normalize(s):
        return " ".join(s.lower().split())

    return normalize(predicted_sql) == normalize(gold_sql)


def execution_match(pred_rows: list, gold_rows: list) -> bool:
    if not pred_rows and not gold_rows:
        return True

    if len(pred_rows) != len(gold_rows):
        return False

    to_str = lambda row: str(sorted(row.items()))
    return sorted(to_str(r) for r in pred_rows) == sorted(to_str(r) for r in gold_rows)