import sqlite3
import logging
import os

logger = logging.getLogger("execution")


def execute_sql(sql: str, db_id: str = None) -> dict:
    if db_id:
        db_path = f"db/{db_id}.db"
    else:
        db_path = None

    if not db_path:
        return {
            "success": False,
            "rows": [],
            "row_count": 0,
            "error": "No database specified (use_retrieved_context must be true to resolve a db_id)",
        }

    if not os.path.exists(db_path):
        return {
            "success": False,
            "rows": [],
            "row_count": 0,
            "error": f"Database file not found: {db_path}",
        }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()
        cursor.execute(sql)
        raw_rows = cursor.fetchmany(100)
        rows = [dict(r) for r in raw_rows]
        conn.close()

        logger.info(f"Executed query on {db_id}, got {len(rows)} rows")
        return {"success": True, "rows": rows, "row_count": len(rows), "error": None}

    except sqlite3.OperationalError as e:
        logger.warning(f"SQL execution failed: {e}")
        return {"success": False, "rows": [], "row_count": 0, "error": str(e)}

    except Exception as e:
        logger.error(f"Unexpected execution error: {e}")
        return {"success": False, "rows": [], "row_count": 0, "error": str(e)}