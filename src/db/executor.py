from typing import Any, Dict, List, Optional

import psycopg2

from src.db.connector import DatabaseConnector


class QueryExecutionError(Exception):
    def __init__(self, message: str, original_exception: Exception):
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"{message}: {original_exception}")


def measure_query(
    db_connector: DatabaseConnector,
    query_sql: str,
    settings: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = db_connector.get_connection()

        with conn.cursor() as cursor:
            # Use BEGIN...ROLLBACK to ensure settings are transaction-scoped
            # and ANALYZE does not have side effects.
            cursor.execute("BEGIN;")

            # 1. Apply session-local settings (e.g., disable hashjoin)
            if settings:
                for setting in settings:
                    local_setting = setting.replace("SET ", "SET LOCAL ")
                    cursor.execute(local_setting)

            # 2. Run 1 (Warm-up):
            cursor.execute(query_sql)
            while cursor.fetchone():  # Consume results
                pass

            # 3. Run 2 (Warm-up):
            cursor.execute(query_sql)
            while cursor.fetchone():  # Consume results
                pass

            # 4. Run 3 (Measurement):
            explain_command = f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS) {query_sql}"
            cursor.execute(explain_command)
            result_json = cursor.fetchone()[0]
            if not result_json:
                raise ValueError("EXPLAIN command returned no result.")

            explain_data = result_json[0]

            # 5. Parse the results
            plan_json = explain_data["Plan"]
            actual_latency_ms = explain_data["Execution Time"]
            estimated_cost = plan_json["Total Cost"]

            # 6. Always rollback
            cursor.execute("ROLLBACK;")

            return {
                "actual_latency_ms": actual_latency_ms,
                "estimated_cost": estimated_cost,
                "plan_json": plan_json,
                "full_explain_json": result_json,
            }

    except (psycopg2.Error, ValueError, KeyError) as e:
        print(f"[Executor Error] Failed to measure query: {e}")
        if conn:
            conn.rollback()
        raise QueryExecutionError("Failed to execute and measure query", e) from e

    finally:
        if conn:
            db_connector.release_connection(conn)


def get_plan_json(
    db_connector: DatabaseConnector,
    query_sql: str,
    settings: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = db_connector.get_connection()

        with conn.cursor() as cursor:
            # Use BEGIN...ROLLBACK to ensure settings are transaction-scoped
            cursor.execute("BEGIN;")

            if settings:
                for setting in settings:
                    local_setting = setting.replace("SET ", "SET LOCAL ")
                    cursor.execute(local_setting)

            # Run EXPLAIN (FORMAT JSON) - no ANALYZE
            explain_command = f"EXPLAIN (FORMAT JSON) {query_sql}"
            cursor.execute(explain_command)

            result_json = cursor.fetchone()[0]
            if not result_json:
                raise ValueError("EXPLAIN command returned no result.")

            explain_data = result_json[0]

            # Always rollback
            cursor.execute("ROLLBACK;")

            return explain_data["Plan"]  # Return only the 'Plan' tree

    except (psycopg2.Error, ValueError, KeyError) as e:
        # A plan might be invalid with certain settings (e.g., all joins off)
        # This is not an error, just an invalid candidate.
        print(f"[Executor Info] Could not get plan for settings: {e}")
        if conn:
            conn.rollback()
        return None  # Return None for invalid plans

    finally:
        if conn:
            db_connector.release_connection(conn)
