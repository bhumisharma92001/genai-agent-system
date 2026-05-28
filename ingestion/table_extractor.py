# ingestion/table_extractor.py

import sqlite3
import uuid

import pdfplumber
import pandas as pd


class TableExtractor:

    def __init__(
        self,
        sqlite_db_path: str = "memory/tables.db"
    ):

        self.connection = sqlite3.connect(
            sqlite_db_path
        )

    def extract(
        self,
        file_path: str
    ):

        if file_path.endswith(".pdf"):
            return self._extract_pdf_tables(
                file_path
            )

        if (
            file_path.endswith(".csv")
            or file_path.endswith(".xlsx")
            or file_path.endswith(".xls")
        ):
            return self._extract_dataframe_tables(
                file_path
            )

        return []

    def _extract_pdf_tables(
        self,
        file_path: str
    ):

        summaries = []

        with pdfplumber.open(file_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                extracted_tables = page.extract_tables()

                for table_index, table in enumerate(
                    extracted_tables
                ):

                    if not table:
                        continue

                    df = pd.DataFrame(
                        table[1:],
                        columns=table[0]
                    )

                    table_name = (
                        f"table_"
                        f"{uuid.uuid4().hex[:8]}"
                    )

                    df.to_sql(
                        table_name,
                        self.connection,
                        if_exists="replace",
                        index=False
                    )

                    summary = self._generate_table_summary(
                        df=df,
                        table_name=table_name,
                        page_number=page_number
                    )

                    summaries.append(summary)

        return summaries

    def _extract_dataframe_tables(
        self,
        file_path: str
    ):

        if file_path.endswith(".csv"):

            df = pd.read_csv(file_path)

        else:

            df = pd.read_excel(file_path)

        table_name = (
            f"table_"
            f"{uuid.uuid4().hex[:8]}"
        )

        df.to_sql(
            table_name,
            self.connection,
            if_exists="replace",
            index=False
        )

        summary = self._generate_table_summary(
            df=df,
            table_name=table_name,
            page_number=1
        )

        return [summary]

    @staticmethod
    def _generate_table_summary(
        df,
        table_name: str,
        page_number: int
    ) -> dict:

        columns = list(df.columns)

        row_count = len(df)

        preview_rows = df.head(3).to_dict(
            orient="records"
        )

        summary_text = (
            f"Table {table_name} contains "
            f"{row_count} rows with columns: "
            f"{', '.join(columns)}. "
            f"Sample rows: {preview_rows}"
        )

        return {
            "text": summary_text,
            "metadata": {
                "table_name": table_name,
                "page": page_number,
                "chunk_type": "table"
            }
        }