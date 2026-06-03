import sqlglot
import sqlglot.errors


def validate_sql(sql: str) -> dict:
    if not sql or not sql.strip():
        return {"is_valid": False, "errors": "SQL is empty"}

    try:
        statements = sqlglot.parse(sql)

        if not statements:
            return {"is_valid": False, "errors": "Could not parse SQL"}

        if len(statements) > 1:
            return {"is_valid": False, "errors": "Only one SQL statement allowed"}

        stmt = statements[0]

        if not isinstance(stmt, sqlglot.expressions.Select):
            kind = type(stmt).__name__
            return {"is_valid": False, "errors": f"Only SELECT is allowed. Got: {kind}"}

        return {"is_valid": True, "errors": None}

    except sqlglot.errors.ParseError as e:
        return {"is_valid": False, "errors": str(e)}

    except Exception as e:
        return {"is_valid": False, "errors": f"Unexpected error: {str(e)}"}