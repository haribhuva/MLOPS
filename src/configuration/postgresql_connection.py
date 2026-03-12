import sys
import psycopg2
import os
from src.logger import configure_logger
from src.exception import MyException
import logging
from src.constants import POSTGRE_SQL_PASSWORD, DATABASE_NAME, USER, HOST

def postgresql_client():
    try:
        logging.info("Attempting to connect to PostgreSQL database...")
        conn = psycopg2.connect(
            dbname = DATABASE_NAME,
            user = USER,
            password = POSTGRE_SQL_PASSWORD,
            host = HOST
        )
        logging.info("Successfully connected to PostgreSQL database.")
        return conn
    except Exception as e:
        logging.error("Failed to connect to PostgreSQL database.")
        raise MyException(e, sys)

if __name__ == "__main__":
    conn = postgresql_client()
    if conn:
        logging.info("Connection object created successfully.")
        conn.close()
        logging.info("Connection closed.")