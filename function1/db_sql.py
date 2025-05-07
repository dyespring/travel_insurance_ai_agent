import psycopg2

conn = psycopg2.connect(
    dbname="insurance_db",
    user="admin",
    password="admin123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
cursor.execute("SELECT version();")
print("PostgreSQL version:", cursor.fetchone())
