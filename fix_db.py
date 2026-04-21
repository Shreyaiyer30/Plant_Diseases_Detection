import sqlite3
import os

db_path = 'instance/plantcure.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE predictions SET status = 'healthy', severity = 'None' WHERE disease LIKE '%healthy%' OR disease = 'Healthy'")
    
    # Keep the max ID (most recent) for duplicates
    c.execute("""
        DELETE FROM predictions 
        WHERE id NOT IN (
            SELECT MAX(id) 
            FROM predictions 
            GROUP BY user_id, disease, filename
        )
    """)
    conn.commit()
    conn.close()
    print('Cleaned up database: updated healthy statuses and removed duplicates.')
else:
    print('Database not found.')
