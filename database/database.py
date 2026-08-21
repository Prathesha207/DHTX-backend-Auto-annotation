from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./sqlite.db"

# Create the engine with standard SQLite configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class for DB sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the Base class for the ORM models
Base = declarative_base()

def ensure_db_schema():
    """Ensure SQLite table columns exist without needing manual migration."""
    import sqlite3
    from pathlib import Path
    db_path = Path(__file__).resolve().parent.parent / "sqlite.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            # --- videos table ---
            cur.execute("PRAGMA table_info(videos)")
            cols = [row[1] for row in cur.fetchall()]
            if cols:
                for col, defn in [
                    ("verdict", "TEXT"),
                    ("output_path", "TEXT"),
                    ("relative_path", "TEXT"),
                    ("last_modified", "INTEGER"),
                    ("processing_started_at", "TEXT"),
                ]:
                    if col not in cols:
                        cur.execute(f"ALTER TABLE videos ADD COLUMN {col} {defn}")

            # --- batches table ---
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='batches'")
            row = cur.fetchone()
            if row and row[0]:
                original_sql = row[0]
                
                # Check if the global constraint exists and needs migration.
                # We check if it exists by checking index_list to see what's actually applied,
                # or checking the original SQL to see if it lacks batch_date in the unique constraint.
                cur.execute("PRAGMA index_list(batches)")
                indexes = cur.fetchall()
                needs_migration = False
                
                for idx in indexes:
                    idx_name = idx[1]
                    idx_unique = idx[2]
                    if idx_unique:
                        cur.execute(f"PRAGMA index_info({idx_name})")
                        cols = [c[2] for c in cur.fetchall()]
                        if set(cols) == {"storage_root_id", "batch_number"}:
                            needs_migration = True
                            break
                            
                # Also fallback to looking at original_sql if index pragma doesn't work as expected
                if not needs_migration and "uq_storage_batch_number" in original_sql:
                     needs_migration = True
                     
                if needs_migration:
                    import re
                    # 1. Take the existing CREATE TABLE SQL verbatim
                    # 2. Replace the old constraint with the new one
                    new_sql = original_sql
                    
                    # Remove the old constraint (both named and unnamed variations)
                    new_sql = re.sub(
                        r'CONSTRAINT\s+\w+\s+UNIQUE\s*\([^)]+\)', 
                        'CONSTRAINT uq_storage_date_batch_number UNIQUE (storage_root_id, batch_date, batch_number)', 
                        new_sql, 
                        flags=re.IGNORECASE
                    )
                    
                    if new_sql == original_sql:
                        new_sql = re.sub(
                            r'UNIQUE\s*\([^)]+\)', 
                            'UNIQUE (storage_root_id, batch_date, batch_number)', 
                            new_sql, 
                            flags=re.IGNORECASE
                        )

                    # 3. Point the CREATE TABLE at a temporary name
                    new_sql = new_sql.replace(
                        'CREATE TABLE batches',
                        'CREATE TABLE batches_migration_tmp',
                        1
                    )

                    # 4. Execute the rebuild
                    try:
                        cur.execute(new_sql)
                        cur.execute(
                            "INSERT OR IGNORE INTO batches_migration_tmp "
                            "SELECT * FROM batches"
                        )
                        cur.execute("DROP TABLE batches")
                        cur.execute("ALTER TABLE batches_migration_tmp RENAME TO batches")
                        conn.commit()
                        print("✅ Successfully migrated batches table to per-date unique constraint.")
                    except Exception as e:
                         print("⚠️ Failed to migrate batches table constraint:", e)

            # Ensure folder_name column exists
            cur.execute("PRAGMA table_info(batches)")
            batch_cols = [row[1] for row in cur.fetchall()]
            if batch_cols and "folder_name" not in batch_cols:
                cur.execute("ALTER TABLE batches ADD COLUMN folder_name TEXT")

            # --- index ---
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_batch_status "
                "ON videos(batch_id, status)"
            )
            conn.commit()
        except Exception as e:
            print("DB Schema migration check warning:", e)
        finally:
            conn.close()


# Dependency for FastAPI to inject DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
