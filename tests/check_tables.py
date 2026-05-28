import sqlite3
import pandas as pd


connection = sqlite3.connect(
    "memory/tables.db"
)

cursor = connection.cursor()

cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
)

tables = cursor.fetchall()

print("\nAVAILABLE TABLES:\n")

for table in tables:

    table_name = table[0]

    print(f"\nTABLE: {table_name}\n")

    df = pd.read_sql_query(
        f"SELECT * FROM {table_name}",
        connection
    )

    print(df)

    print("\n" + "=" * 100 + "\n")