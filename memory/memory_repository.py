import sqlite3
import threading

class MemoryRepository:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def initialize(self) -> None:
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
                    facts TEXT,
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

            conn.commit()

    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        with self._lock, self._get_connection() as conn:
            # upsert last session for the user so we can resume by user_id
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
            conn.commit()

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
                INSERT INTO interactions(
                    user_id,
                    session_id,
                    query,
                    answer,
                    importance_score
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    query,
                    answer,
                    importance_score,
                )
            )
            conn.commit()

    def get_recent_interactions(
        self,
        user_id: str,
        session_id: str,
        limit: int
    ) -> list[tuple[str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT query, answer
                FROM interactions
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit)
            ).fetchall()

        return rows

    def get_relevant_interactions(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> list[tuple[str, str]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT query, answer
                FROM interactions
                WHERE user_id = ? AND session_id = ? AND importance_score > 0
                ORDER BY importance_score DESC, id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit)
            ).fetchall()

        return rows

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        facts: str
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries(
                    user_id,
                    session_id,
                    summary,
                    facts
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    summary,
                    facts,
                )
            )
            conn.commit()

    def get_summaries(self, user_id: str, session_id: str):
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT summary, facts
                FROM memory_summaries
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                """,
                (user_id, session_id)
            ).fetchall()

        return rows
