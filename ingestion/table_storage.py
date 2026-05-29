from pathlib import Path
import sqlite3
import uuid

import pandas as pd


class TableStorage:

    def __init__(
        self,
        database_path: str = (
            "memory/tables.db"
        )
    ):

        self.database_path = (
            database_path
        )

        Path(
            self.database_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def store(
        self,
        dataframe: pd.DataFrame
    ) -> str:

        if dataframe.empty:

            raise ValueError(
                "Dataframe is empty"
            )

        table_name = (
            f"table_"
            f"{uuid.uuid4().hex[:8]}"
        )

        try:

            connection = sqlite3.connect(
                self.database_path
            )

            dataframe.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

            connection.close()

            return table_name

        except Exception as e:

            raise RuntimeError(
                "Failed to store table"
            ) from e