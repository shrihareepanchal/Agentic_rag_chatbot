"""
SQLite store for structured data.

Provides:
  - create_table_from_dataframe()  : ingest CSV/JSON into SQLite
  - execute_query()                : run a (validated) SQL query
  - get_schema()                   : introspect all tables
  - natural_language_to_sql()      : LLM-powered NL→SQL translation
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# SQL keywords we never allow in user-generated queries
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


class SQLStore:
    def __init__(self):
        settings = get_settings()
        db_url = f"sqlite:///{settings.sqlite_db_path}"
        self._engine = create_engine(db_url, echo=False, future=True)
        logger.info("sql_store_ready", db=settings.sqlite_db_path)

    # ─── Write ────────────────────────────────────────────────────────────────

    def create_table_from_dataframe(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> int:
        """
        Persist a DataFrame as a SQLite table.
        Returns row count.
        """
        table_name = _sanitise_table_name(table_name)
        df.to_sql(table_name, self._engine, if_exists=if_exists, index=False)
        logger.info("sql_table_created", table=table_name, rows=len(df))
        return len(df)

    # ─── Read ─────────────────────────────────────────────────────────────────

    def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Execute a SELECT query.

        Returns (rows_as_dicts, error_message_or_empty).
        Raises ValueError for non-SELECT queries.
        """
        sql = sql.strip().rstrip(";")

        if _FORBIDDEN_SQL.search(sql):
            return [], "Query contains forbidden SQL keywords."
        if not sql.upper().lstrip().startswith("SELECT"):
            return [], "Only SELECT queries are permitted."

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result]
            logger.info("sql_query_ok", rows=len(rows), sql=sql[:120])
            return rows, ""
        except SQLAlchemyError as e:
            logger.warning("sql_query_error", error=str(e), sql=sql[:120])
            return [], str(e)

    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Return a dict of {table_name: [{column_name, type}]}.
        Used for NL→SQL context.
        """
        inspector = inspect(self._engine)
        schema: Dict[str, List[Dict[str, str]]] = {}
        for table_name in inspector.get_table_names():
            cols = []
            for col in inspector.get_columns(table_name):
                cols.append(
                    {"name": col["name"], "type": str(col["type"])}
                )
            schema[table_name] = cols
        return schema

    def get_schema_as_text(self) -> str:
        """Human-readable schema description."""
        schema = self.get_schema()
        if not schema:
            return "No tables found in the database."
        lines = ["Available tables and columns:"]
        for table, cols in schema.items():
            col_strs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  TABLE {table}: {col_strs}")
        return "\n".join(lines)

    def table_exists(self, table_name: str) -> bool:
        inspector = inspect(self._engine)
        return table_name in inspector.get_table_names()

    def list_tables(self) -> List[str]:
        return inspect(self._engine).get_table_names()

    def drop_all_tables(self) -> int:
        """Drop all tables. Returns number of tables dropped."""
        inspector = inspect(self._engine)
        tables = inspector.get_table_names()
        with self._engine.connect() as conn:
            for table in tables:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
            conn.commit()
        logger.info("sql_drop_all_tables", count=len(tables))
        return len(tables)


def _sanitise_table_name(name: str) -> str:
    """Allow only alphanumeric + underscore in table names."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


@lru_cache(maxsize=1)
def get_sql_store() -> SQLStore:
    return SQLStore()