import sqlite3, os, json
db_path = 'instance/plantcure.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM plant_treatments").fetchall()
    for r in rows:
        print(f"Disease: {r['disease_name']}")
        print(f"Treatment: {r['treatment_steps'][:50]}...")
    conn.close()
else:
    print("DB not found")
