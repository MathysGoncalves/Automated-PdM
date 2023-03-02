import os
import psycopg2

conn = psycopg2.connect(
        host="localhost",
        database="flask_db",
        user="postgres",
        password="oui")

# Open a cursor to perform database operations
cur = conn.cursor()

conn.commit()

cur.close()
conn.close()