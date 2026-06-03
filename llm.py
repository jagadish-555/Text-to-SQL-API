import os
import re
import time
import logging
import sqlglot
import sqlglot.expressions as sqle
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

FEW_SHOT_EXAMPLES = """\
Question: How many courses are offered by the Mathematics department?
SQL: SELECT COUNT(*) AS course_count FROM CIS_COURSE_CATALOG WHERE SUBJECT_CODE = 18;

Question: What is the department name and school name for each degree-granting department?
SQL: SELECT DEPARTMENT_NAME, SCHOOL_NAME FROM SIS_DEPARTMENT WHERE IS_DEGREE_GRANTING = 'Y';

Question: For each school, how many departments does it have?
SQL: SELECT SCHOOL_NAME, COUNT(*) AS department_count FROM SIS_DEPARTMENT GROUP BY SCHOOL_NAME ORDER BY department_count DESC;

Question: Which departments have more than 10 courses in the catalog?
SQL: WITH dept_counts AS (SELECT SUBJECT_CODE, COUNT(*) AS course_count FROM CIS_COURSE_CATALOG GROUP BY SUBJECT_CODE) SELECT sd.DEPARTMENT_NAME, dc.course_count FROM SIS_DEPARTMENT sd JOIN dept_counts dc ON sd.DEPARTMENT_CODE = dc.SUBJECT_CODE WHERE dc.course_count > 10 ORDER BY dc.course_count DESC;

Question: List courses with the number of enrolled students and the responsible faculty.
SQL: SELECT lso.SUBJECT_CODE, lso.NUM_ENROLLED_STUDENTS, lso.RESPONSIBLE_FACULTY_NAME FROM LIBRARY_SUBJECT_OFFERED lso ORDER BY lso.NUM_ENROLLED_STUDENTS DESC;
"""

SYSTEM_INSTRUCTION = (
    "You are an expert SQL analyst for an enterprise data warehouse. "
    "You MUST output ONLY the raw SQL query with zero additional text, "
    "zero explanation, zero markdown, zero backticks, and zero comments. "
    "The very first character of your response must be S (for SELECT) or W (for WITH)."
)


def build_prompt(question: str, retrieved_tables: list) -> str:
    schema_lines = []
    for t in retrieved_tables:
        col_details = t.get("column_details", [])
        if col_details:
            cols = ", ".join(
                f"{cd['name']} ({cd['type']})" for cd in col_details
            )
        else:
            cols = ", ".join(t["columns"])
        schema_lines.append(f"  {t['table_name']}({cols})")

    schema_block = "\n".join(schema_lines) if schema_lines else "  (no tables retrieved)"

    prompt = f"""You are an expert SQL analyst for an enterprise data warehouse (SQLite dialect).
Output ONLY the raw SQL query. No explanation, no markdown, no backticks, no comments, no prefix.
Start your response directly with SELECT or WITH.

Rules:
- Use only SELECT. Never INSERT, UPDATE, DELETE, DROP.
- CRITICAL: Use ONLY the exact column names listed in the Schema section. Never invent, guess, or infer column names.
- In any query with a JOIN, EVERY column reference MUST be prefixed with its table alias (e.g. t.COLUMN_NAME, never bare COLUMN_NAME) to avoid ambiguous-column errors.
- For complex queries with aggregation + filtering, use a CTE (WITH clause).
- Do not use ROUND() unless the question explicitly requests rounding.
- SQLite has no STDDEV() or VARIANCE(). Use SQRT(AVG(col*col)-AVG(col)*AVG(col)) for standard deviation and AVG(col*col)-AVG(col)*AVG(col) for variance.
- For geometric mean use EXP(AVG(LN(col))).
- Filter years/terms using LIKE '2019%' or = 2019 depending on column type.

Schema:
{schema_block}

Examples:
{FEW_SHOT_EXAMPLES}
Question: {question}
SQL:"""

    return prompt


def call_groq(prompt: str, max_retries: int = 3) -> dict:
    last_error = None
    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    for model in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content
                logger.debug(f"LLM attempt {attempt} succeeded with {model}")
                return {"raw": raw, "error": None}

            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
                logger.warning(f"LLM attempt {attempt} failed with {model}: {e}")
                if "rate_limit" in err_msg.lower() or "429" in err_msg:
                    break
                if attempt < max_retries:
                    wait = 2 ** (attempt - 1)
                    time.sleep(wait)

    logger.error(f"All LLM attempts failed: {last_error}")
    return {"raw": None, "error": last_error}


def _is_complete_sql(sql: str) -> bool:
    if not sql:
        return False
    upper = sql.upper().strip()
    if sql.count("(") != sql.count(")"):
        return False
    if upper.endswith(("SELECT", "FROM", "WHERE", "JOIN", "ON",
                       "AND", "OR", "GROUP BY", "ORDER BY", "HAVING",
                       "WITH", "AS", "SET", "INTO")):
        return False
    return True


def _hallucinated_cols(sql: str, retrieved_tables: list) -> list:
    if not sql or not retrieved_tables:
        return []
    valid: set = set()
    for t in retrieved_tables:
        for cd in t.get("column_details", []):
            valid.add(cd["name"].upper())
        for c in t.get("columns", []):
            valid.add(c.upper())
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    introduced: set = set()
    for node in tree.find_all(sqle.Alias):
        if node.alias:
            introduced.add(node.alias.upper())
    bad, seen = [], set()
    for col in tree.find_all(sqle.Column):
        name = col.name.upper()
        if not name or name == "*" or name in seen:
            continue
        seen.add(name)
        if name not in valid and name not in introduced:
            bad.append(col.name)
    return bad


def clean_sql(raw: str) -> str:
    if not raw:
        return ""

    sql = raw.strip()
    sql = sql.replace("```sql", "").replace("```SQL", "").replace("```", "")

    if sql.upper().startswith("SQL:"):
        sql = sql[4:]

    sql = sql.strip()

    lines = sql.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith("SELECT") or stripped.startswith("WITH"):
            start_idx = i
            break

    sql = "\n".join(lines[start_idx:]).strip()

    end_idx = len(sql)
    for match in re.finditer(r';', sql):
        candidate = sql[:match.start()].strip()
        open_p = candidate.count("(") - candidate.count(")")
        if open_p == 0:
            end_idx = match.start()
            break

    sql = sql[:end_idx].strip()

    return sql


def generate_sql(question: str, retrieved_tables: list) -> dict:
    prompt = build_prompt(question, retrieved_tables)
    result = call_groq(prompt)

    if result["error"]:
        return {"sql": None, "prompt_used": prompt, "error": result["error"]}

    sql = clean_sql(result["raw"])

    if not _is_complete_sql(sql):
        logger.warning("Incomplete SQL detected, retrying")
        r2 = call_groq(prompt)
        if not r2["error"]:
            s2 = clean_sql(r2["raw"])
            if _is_complete_sql(s2):
                sql = s2

    bad = _hallucinated_cols(sql, retrieved_tables)
    if bad:
        logger.warning(f"Hallucinated columns detected: {bad}. Retrying with correction hint.")
        fix_prompt = (
            prompt
            + f"\n\nWARNING: The column(s) {bad} do not exist in the schema above."
            " Rewrite the query using ONLY the exact column names from the Schema section."
        )
        r3 = call_groq(fix_prompt)
        if not r3["error"]:
            s3 = clean_sql(r3["raw"])
            if _is_complete_sql(s3) and not _hallucinated_cols(s3, retrieved_tables):
                sql = s3

    return {"sql": sql, "prompt_used": prompt, "error": None}
