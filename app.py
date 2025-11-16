import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.db.connector import DatabaseConnector
from src.db.executor import QueryExecutionError, measure_query
from src.optimizers import PostgresGEQOOptimizer, PostgresOptimizer, XGBoostLQO
from src.optimizers.base import BaseOptimizer

st.set_page_config(page_title="Project Lighthouse", page_icon="", layout="wide")


@st.cache_resource
def load_optimizers() -> List[BaseOptimizer]:
    print("--- Caching: Initializing Optimizers... ---")

    pg_default = PostgresOptimizer()
    pg_geqo = PostgresGEQOOptimizer(geqo_threshold=2)

    lqo = XGBoostLQO()
    try:
        lqo.load_model()
    except FileNotFoundError:
        st.error(
            "Fatal Error: Model files not found in /models directory. "
            "Please run 'train_lqo.py' first."
        )
        return None

    return [pg_default, pg_geqo, lqo]


def run_comparison(
    query_sql: str, optimizers: List[BaseOptimizer]
) -> List[Dict[str, Any]]:
    db_conn = None
    results = []

    try:
        db_conn = DatabaseConnector()

        for optimizer in optimizers:
            plan_dict = optimizer.get_plan(query_sql)

            measurement = measure_query(
                db_connector=db_conn,
                query_sql=plan_dict["plan_sql"],
                settings=plan_dict["settings"],
            )

            if measurement:
                total_time = (
                    plan_dict["inference_time_ms"] + measurement["actual_latency_ms"]
                )
                result_row = {
                    "Optimizer": optimizer.optimizer_name,
                    "Inference (ms)": plan_dict["inference_time_ms"],
                    "Execution (ms)": measurement["actual_latency_ms"],
                    "Total Time (ms)": total_time,
                    "Estimated Cost": measurement["estimated_cost"],
                    "full_explain_json": measurement["full_explain_json"],
                }
                results.append(result_row)

    except QueryExecutionError as e:
        st.error(f"Error during execution: {e.message}")
    except Exception as e:
        st.exception(e)
    finally:
        if db_conn:
            db_conn.close_all()

    return results


st.title("Project Lighthouse")
st.header("Interactive Query Optimizer Comparison")

optimizers = load_optimizers()
if optimizers:
    DEFAULT_QUERY = (
        "SELECT * FROM title t "
        "JOIN cast_info ci ON t.id = ci.movie_id "
        "WHERE t.production_year > 2005 AND ci.role_id = 1;"
    )

    sql_query = st.text_area("Enter your SQL query:", DEFAULT_QUERY, height=150)

    if st.button("Compare Plans"):
        if not sql_query.strip():
            st.error("Query cannot be empty.")
        else:
            with st.spinner("Running optimizers... (This may take a few seconds)"):
                comparison_results = run_comparison(sql_query, optimizers)

            if comparison_results:
                st.subheader("Performance Summary")

                df = pd.DataFrame(comparison_results)
                df_display = df.drop(columns=["full_explain_json"])

                st.dataframe(
                    df_display.style.format(
                        {
                            "Inference (ms)": "{:.2f}",
                            "Execution (ms)": "{:.2f}",
                            "Total Time (ms)": "{:.2f}",
                            "Estimated Cost": "{:.2f}",
                        }
                    ).highlight_min(subset=["Total Time (ms)"], color="lightgreen")
                )

                st.subheader("Plan Details")
                st.info(
                    "You can copy the JSON from any plan and visualize it at [https://explain.dalibo.com/](https://explain.dalibo.com/)"
                )

                for res in comparison_results:
                    with st.expander(f"Show Plan for: **{res['Optimizer']}**"):
                        st.json(res["full_explain_json"])
            else:
                st.error("Failed to get results for this query.")
