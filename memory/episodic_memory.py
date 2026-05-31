import sqlite3

from memory.base_memory import BaseMemory


class EpisodicMemory(BaseMemory):

    def __init__(
        self,
        db_path: str = "memory/memory.db"
    ):
        self.db_path = db_path
        self._initialize()

    def _initialize(self) -> None:

        with sqlite3.connect(self.db_path) as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_summaries(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    facts TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

    def save_interaction(
        self,
        query: str,
        answer: str
    ) -> None:

        with sqlite3.connect(self.db_path) as conn:

            conn.execute(
                """
                INSERT INTO interactions(
                    query,
                    answer
                )
                VALUES (?, ?)
                """,
                (
                    query,
                    answer
                )
            )

            conn.commit()

    def get_recent_interactions(
        self,
        limit: int = 5
    ):

        with sqlite3.connect(self.db_path) as conn:

            rows = conn.execute(
                """
                SELECT
                    query,
                    answer
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return rows

    def get_summaries(self)->list:

        with sqlite3.connect(self.db_path) as conn:

            rows = conn.execute(
                """
                SELECT summary, facts
                FROM memory_summaries
                ORDER BY id DESC
                """
            ).fetchall()

        return rows

    def save_summary(
        self,
        summary: str,
        facts: str
    ) -> None:

        with sqlite3.connect(self.db_path) as conn:

            conn.execute(
                """
                INSERT INTO memory_summaries(
                    summary,
                    facts
                )
                VALUES (?, ?)
                """,
                (
                    summary,
                    facts
                )
            )

            conn.commit()