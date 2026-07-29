import sqlite3
conn = sqlite3.connect('D:/Dhtx-auto-annotation/backend/app/database/app.db')
cursor = conn.cursor()
cursor.execute("SELECT message FROM logs ORDER BY id DESC LIMIT 10")
for row in cursor.fetchall():
    print(row[0])
