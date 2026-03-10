import sys
import psycopg2
from dotenv import load_dotenv
import os
from src.logger import configure_logger
from src.exception import MyException
import logging

load_dotenv()

def connect_to_postgresql():
    try:
        logging.info("Attempting to connect to PostgreSQL database...")
        conn = psycopg2.connect(
            dbname="mlops",
            user="postgres",
            password=os.getenv("POSTGRE_SQL_PASSWORD"),
            host="localhost"
        )
        logging.info("Successfully connected to PostgreSQL database.")
        return conn
    except Exception as e:
        logging.error("Failed to connect to PostgreSQL database.")
        raise MyException(e, sys)

if __name__ == "__main__":
    conn = connect_to_postgresql()
    if conn:
        logging.info("Connection object created successfully.")
        conn.close()
        logging.info("Connection closed.")