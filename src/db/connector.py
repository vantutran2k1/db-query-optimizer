from __future__ import annotations

import os
from typing import Optional, Any

import psycopg2
import psycopg2.extras
import yaml
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

load_dotenv()


class DatabaseConnector:
    _instance: Optional[DatabaseConnector] = None
    _pool: Optional[SimpleConnectionPool] = None

    def __new__(cls) -> DatabaseConnector:
        if cls._instance is None:
            cls._instance = super(DatabaseConnector, cls).__new__(cls)
            cls._instance._init_config()
            cls._instance._init_pool()

        return cls._instance

    def get_connection(self) -> Any:
        if not self._pool:
            raise RuntimeError("Connection pool is not initialized")

        conn = self._pool.getconn()

        psycopg2.extras.register_default_jsonb(conn_or_curs=conn, globally=False)

        settings = self._yaml_config.get("connection", {})
        if settings:
            with conn.cursor() as cursor:
                for key, value in settings.items():
                    cursor.execute(f"SET {key} = %s;", (value,))
            conn.commit()

        return conn

    def release_connection(self, conn: Any):
        if self._pool:
            self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None
        DatabaseConnector._instance = None
        print("Database connection pool closed.")

    def _init_config(self):
        self._db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }

        for param, value in self._db_params.items():
            if not value:
                raise ValueError(
                    f"Missing database connection parameter: {param.upper()}"
                    "Check your .env file"
                )

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "db.yml"
        )
        try:
            with open(config_path, "r") as f:
                self._yaml_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {config_path} not found. Using defaults.")
            self._yaml_config = {"connection": {}}

    def _init_pool(self):
        try:
            self._pool = SimpleConnectionPool(minconn=1, maxconn=10, **self._db_params)
        except psycopg2.OperationalError as e:
            print(f"FATAL: Could not connect to database: {e}")
            raise e
