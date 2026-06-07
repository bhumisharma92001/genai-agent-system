import sqlite3
import threading
from contextlib import contextmanager

class MemoryRepository:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

    @contextmanager
    def _get_connection(self):
        """
        Custom context manager that handles proper connection cleanup 
        AND native SQLite transaction rollback/commit lifecycles.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        # Transactions are handled automatically by the custom context manager
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    importance_score INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_summaries(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions(
                    user_id TEXT PRIMARY KEY,
                    last_session_id TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions(user_id, last_session_id, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_session_id = excluded.last_session_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, session_id)
            )

    def get_last_session_id(self, user_id: str) -> str | None:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT last_session_id
                FROM user_sessions
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,)
            ).fetchone()
        return row[0] if row else None

    def save_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        importance_score: int = 1
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO interactions(user_id, session_id, query, answer, importance_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, session_id, query, answer, importance_score)
            )

    def get_recent_interactions(
        self,
        user_id: str,
        session_id: str,
        limit: int
    ) -> list[tuple[str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT query, answer FROM (
                    SELECT id, query, answer
                    FROM interactions
                    WHERE user_id = ? AND session_id = ? AND importance_score > 0
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (user_id, session_id, limit)
            ).fetchall()
        return rows

    def get_interactions_after_timestamp(
        self,
        user_id: str,
        session_id: str,
        timestamp: str
    ) -> list[tuple[str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT query, answer FROM interactions
                WHERE user_id = ? AND session_id = ? AND created_at > ? AND importance_score > 0
                ORDER BY id ASC
                """,
                (user_id, session_id, timestamp)
            ).fetchall()
        return rows

    def get_relevant_interactions(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> list[tuple[str, str]]:
        keywords = [f"%{word}%" for word in query.lower().split() if len(word) > 3]
        with self._lock, self._get_connection() as conn:
            if keywords:
                like_clauses = " OR ".join(["LOWER(query) LIKE ?" for _ in keywords])
                sql = f"""
                    SELECT query, answer FROM (
                        SELECT id, query, answer
                        FROM interactions
                        WHERE user_id = ? AND session_id = ? AND importance_score > 0
                        AND ({like_clauses})
                        ORDER BY importance_score DESC, id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                """
                rows = conn.execute(sql, (user_id, session_id, *keywords, limit)).fetchall()
                if rows:
                    return rows

            rows = conn.execute(
                """
                SELECT query, answer FROM (
                    SELECT id, query, answer
                    FROM interactions
                    WHERE user_id = ? AND session_id = ? AND importance_score > 0
                    ORDER BY importance_score DESC, id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (user_id, session_id, limit)
            ).fetchall()
        return rows

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries(user_id, session_id, summary)
                VALUES (?, ?, ?)
                """,
                (user_id, session_id, summary)
            )

    def get_summaries(self, user_id: str, session_id: str) -> list[tuple[str, str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT summary, created_at
                FROM memory_summaries
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                """,
                (user_id, session_id)
            ).fetchall()
        return rows