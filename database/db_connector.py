
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def get_postgres_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def create_queries_table_if_not_exists(conn):
    """Creates the queries table if it doesn't already exist."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                rag_response TEXT,
                gpt_response TEXT,
                sentiment_score JSONB,
                recommendation JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print(" Table 'queries' checked/created successfully.")

if __name__ == "__main__":
    try:
        conn = get_postgres_connection()
        print(" Connected to PostgreSQL")

        # Create table
        create_queries_table_if_not_exists(conn)

        conn.close()
    except Exception as e:
        print(" PostgreSQL connection failed:", e)