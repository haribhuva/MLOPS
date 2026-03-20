"""PostgreSQL connection factory using environment credentials."""
import sys

import psycopg2

from src.constants import POSTGRE_SQL_PASSWORD, DATABASE_NAME, USER, HOST
from src.exception import MyException
from src.logger import logging


class PostgreSQLClient:
    """Creates and returns a psycopg2 connection to the project database."""

    @staticmethod
    def connect() -> psycopg2.extensions.connection:
        """Open a new connection to the PostgreSQL database."""
        try:
            logging.info("Attempting to connect to PostgreSQL database.")
            conn = psycopg2.connect(
                dbname=DATABASE_NAME,
                user=USER,
                password=POSTGRE_SQL_PASSWORD,
                host=HOST,
            )
            logging.info("Successfully connected to PostgreSQL database.")
            return conn
        except Exception as e:
            logging.error("Failed to connect to PostgreSQL database.")
            raise MyException(e, sys) from e


# Backward-compatible alias used by data_ingestion
postgresql_client = PostgreSQLClient.connect