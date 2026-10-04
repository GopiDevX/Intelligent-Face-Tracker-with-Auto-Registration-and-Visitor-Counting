import sqlite3
import os
from datetime import datetime
import json

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for better concurrent write performance and reliability
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _initialize_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Visitors Table (Registry)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id TEXT PRIMARY KEY,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                embedding TEXT,
                total_visits INTEGER DEFAULT 1
            )
        ''')
        
        # Events Table (Logs Entry/Exits)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                event_type TEXT,
                timestamp TIMESTAMP,
                image_path TEXT,
                FOREIGN KEY(visitor_id) REFERENCES visitors(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def register_visitor(self, visitor_id, embedding, timestamp):
        conn = self._get_connection()
        cursor = conn.cursor()
        # Store embedding as JSON string for simple persistence
        emb_str = json.dumps(embedding.tolist())
        cursor.execute('''
            INSERT INTO visitors (id, first_seen, last_seen, embedding)
            VALUES (?, ?, ?, ?)
        ''', (visitor_id, timestamp, timestamp, emb_str))
        conn.commit()
        conn.close()

    def log_event(self, visitor_id, event_type, timestamp, image_path):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (visitor_id, event_type, timestamp, image_path)
            VALUES (?, ?, ?, ?)
        ''', (visitor_id, event_type, timestamp, image_path))
        conn.commit()
        conn.close()

    def update_visitor_last_seen(self, visitor_id, timestamp):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE visitors SET last_seen = ? WHERE id = ?
        ''', (timestamp, visitor_id))
        conn.commit()
        conn.close()

    def get_all_registered_embeddings(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, embedding FROM visitors')
        rows = cursor.fetchall()
        conn.close()
        
        registered_faces = {}
        for row in rows:
            visitor_id, emb_str = row
            registered_faces[visitor_id] = json.loads(emb_str)
        return registered_faces

    def get_unique_visitor_count(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM visitors')
        count = cursor.fetchone()[0]
        conn.close()
        return count
