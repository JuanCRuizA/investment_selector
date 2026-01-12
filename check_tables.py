import sqlite3
import pandas as pd

conn = sqlite3.connect('data/trading_data.db')

# Ver tablas
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tablas en la base de datos:")
for row in cursor.fetchall():
    print(f"  - {row[0]}")

# Ver columnas de prices_daily
print("\nColumnas de prices_daily:")
df = pd.read_sql("SELECT * FROM prices_daily LIMIT 5", conn)
print(df.columns.tolist())
print("\nPrimeras filas:")
print(df)

conn.close()
