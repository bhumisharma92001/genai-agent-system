import sqlite3
import threading
from contextlib import contextmanager
from utils.logger import logger
from exceptions.custom_errors import EpisodicMemoryError


class MemoryRepository:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

    @contextmanager
    def _get_connection(self):
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
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interactions(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        importance_score INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_summaries(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        last_summarized_id INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, session_id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        except Exception as e:
            logger.error(f"DB initialization failed: {str(e)}")
            raise EpisodicMemoryError(f"Failed to initialize memory DB: {e}") from e

    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions(user_id, session_id, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, session_id)
            )

    def get_all_sessions(self, user_id: str) -> list[tuple[str, str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, created_at, updated_at
                FROM user_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,)
            ).fetchall()
        return rows

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

    def get_interactions_after_id(self, user_id: str, session_id: str, last_id: int) -> list[tuple[str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT query, answer FROM interactions
                WHERE user_id = ? AND session_id = ?
                AND id > ? AND importance_score > 0
                ORDER BY id ASC
                """,
                (user_id, session_id, last_id)
            ).fetchall()
        return rows

    def get_max_interaction_id(self, user_id: str, session_id: str) -> int:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(id) FROM interactions
                WHERE user_id = ? AND session_id = ? AND importance_score > 0
                """,
                (user_id, session_id)
            ).fetchone()
        return row[0] if row and row[0] is not None else 0

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
                like_clauses = " OR ".join(["LOWER(query) LIKE ? OR LOWER(answer) LIKE ?" for _ in keywords])
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
                doubled_keywords = [kw for kw in keywords for _ in range(2)]
                rows = conn.execute(sql, (user_id, session_id, *doubled_keywords, limit)).fetchall()
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

    def save_summary(self, user_id: str, session_id: str, summary: str, last_summarized_id: int) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries(user_id, session_id, summary, last_summarized_id, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    summary = excluded.summary,
                    last_summarized_id = excluded.last_summarized_id,
                    created_at = CURRENT_TIMESTAMP
                """,
                (user_id, session_id, summary, last_summarized_id)
            )

    def get_summaries(self, user_id: str, session_id: str) -> list[tuple[str, int]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT summary, last_summarized_id
                FROM memory_summaries
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                """,
                (user_id, session_id)
            ).fetchall()
        return rows