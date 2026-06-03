import json
import os
import sqlite3

from datasets import load_dataset
from dotenv import load_dotenv
import math

load_dotenv()


def load_beaver_tables():
    print("Loading beaverbench/beaver-table...")
    data = load_dataset("beaverbench/beaver-table", token=os.environ.get("HF_TOKEN"))

    json_fields = ["column_names", "column_types", "example_rows", "example_columns"]
    all_tables = []

    for split in data.keys():
        for row in data[split]:
            parsed = {}
            for k, v in row.items():
                if k in json_fields and isinstance(v, str) and v:
                    try:
                        parsed[k] = json.loads(v)
                    except json.JSONDecodeError:
                        parsed[k] = []
                else:
                    parsed[k] = v
            all_tables.append(parsed)

    print(f"  {len(all_tables)} tables across {len(data.keys())} domain splits\n")
    return all_tables


def load_beaver_queries():
    print("Loading beaverbench/beaver-query...")
    data = load_dataset("beaverbench/beaver-query", token=os.environ.get("HF_TOKEN"))

    json_fields = [
        "tables", "join_keys", "column_mapping",
        "domain_knowledge", "sub_questions", "sub_sqls",
    ]
    all_queries = []

    for split in data.keys():
        for row in data[split]:
            parsed = {}
            for k, v in row.items():
                if k in json_fields and isinstance(v, str) and v:
                    try:
                        parsed[k] = json.loads(v)
                    except json.JSONDecodeError:
                        parsed[k] = []
                else:
                    parsed[k] = v
            all_queries.append(parsed)

    print(f"  {len(all_queries)} queries across {len(data.keys())} domain splits\n")
    return all_queries



def extract_schemas(tables):
    schemas = {}

    for row in tables:
        db_id = row.get("db", "")
        table_name = row.get("table_name", "")
        column_names = row.get("column_names", [])
        column_types = row.get("column_types", [])

        if not db_id or not table_name or not column_names:
            continue

        key = f"{db_id}.{table_name}"

        cols = []
        for i, col_name in enumerate(column_names):
            col_type = column_types[i] if i < len(column_types) else "text"
            cols.append({"name": col_name, "type": col_type})

        col_text = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        description = f"Table {table_name} in database {db_id}. Columns: {col_text}"

        schemas[key] = {
            "db_id": db_id,
            "table_name": table_name,
            "columns": [c["name"] for c in cols],
            "column_details": cols,
            "description": description,
        }

    return schemas


def extract_benchmark(queries, count=25):
    picked = []
    seen = set()

    sorted_queries = sorted(
        queries,
        key=lambda r: 0 if "complex" in r.get("category", "") else 1,
    )

    for row in sorted_queries:
        if len(picked) >= count:
            break

        question = row.get("question", "")
        sql = row.get("sql", "")
        db_id = row.get("db", "")
        tables = row.get("tables", [])

        if not question or not sql or not db_id:
            continue

        if question in seen:
            continue
        seen.add(question)

        picked.append({
            "question": question,
            "gold_sql": sql,
            "db_id": db_id,
            "gold_tables": tables if isinstance(tables, list) else [],
        })

    return picked


def create_databases(tables):
    os.makedirs("db", exist_ok=True)

    connections = {}
    total_tables = 0
    total_rows = 0

    for row in tables:
        db_id = row.get("db", "")
        table_name = row.get("table_name", "")
        column_names = row.get("column_names", [])
        example_rows = row.get("example_rows", [])

        if not db_id or not table_name or not column_names:
            continue

        if db_id not in connections:
            connections[db_id] = sqlite3.connect(f"db/{db_id}.db")

        conn = connections[db_id]
        cursor = conn.cursor()

        col_defs = ", ".join(f'"{c}" TEXT' for c in column_names)
        create_stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})'

        try:
            cursor.execute(create_stmt)
            total_tables += 1
        except sqlite3.Error as e:
            print(f"  Could not create {db_id}.{table_name}: {e}")
            continue

        if example_rows and isinstance(example_rows, list):
            placeholders = ", ".join("?" * len(column_names))
            insert_stmt = f'INSERT OR IGNORE INTO "{table_name}" VALUES ({placeholders})'
            for data_row in example_rows:
                if isinstance(data_row, (list, tuple)) and len(data_row) == len(column_names):
                    try:
                        values = [None if (isinstance(v, float) and math.isnan(v)) else (str(v) if v is not None else None) for v in data_row]
                        cursor.execute(insert_stmt, values)
                        total_rows += 1
                    except sqlite3.Error:
                        pass

        conn.commit()

    for conn in connections.values():
        conn.close()

    print(
        f"  {len(connections)} database files, "
        f"{total_tables} tables, {total_rows} sample rows inserted"
    )



if __name__ == "__main__":
    tables = load_beaver_tables()
    queries = load_beaver_queries()

    print("Building schemas.json...")
    schemas = extract_schemas(tables)
    print(f"  {len(schemas)} unique tables\n")

    print("Building benchmark.json...")
    benchmark = extract_benchmark(queries, count=25)
    print(f"  {len(benchmark)} benchmark questions\n")

    os.makedirs("data", exist_ok=True)
    with open("data/schemas.json", "w") as f:
        json.dump(schemas, f, indent=2)
    print("Saved data/schemas.json")

    with open("data/benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print("Saved data/benchmark.json\n")

    print("Creating SQLite databases...")
    create_databases(tables)

    print("\nAll done.")
    print("  data/schemas.json")
    print("  data/benchmark.json")
    print("  db/{db_id}.db  (one per database domain)")