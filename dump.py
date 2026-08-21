import sqlite3
conn = sqlite3.connect('c:/Users/EmageVision/OneDrive/Desktop/DHTX-NEW/backend/sqlite.db')
c = conn.cursor()
c.execute('SELECT id, batch_id, filename, status FROM videos')
for row in c.fetchall():
    print(row)
conn.close()
