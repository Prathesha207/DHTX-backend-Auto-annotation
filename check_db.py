import sqlite3
conn = sqlite3.connect('D:/Dhtx-auto-annotation/backend/app/database/app.db')
cursor = conn.cursor()
cursor.execute("SELECT id, status, started_at, completed_at FROM video_runs ORDER BY id DESC LIMIT 5")
for row in cursor.fetchall():
    print(row)
