import pandas as pd
from utils.logger import logger
from exceptions.custom_errors import ExtractionError, InvalidInputError


class TableSummarizer:

    def summarize(self, dataframe: pd.DataFrame, table_title: str = "") -> str:
        """
        Produce a rich natural-language summary of the full table.
        Includes: title, columns, numeric stats, and all row data.
        This is stored as the 'table' chunk — used for 'summarize the table' queries.
        """
        if dataframe is None or dataframe.empty:
            raise InvalidInputError("Dataframe is empty")

        try:
            columns = list(dataframe.columns)
            row_count = len(dataframe)
            title_line = f"Table: {table_title}\n" if table_title else ""

            parts = [
                f"{title_line}"
                f"This table has {row_count} rows and {len(columns)} columns: "
                f"{', '.join(columns)}.\n"
            ]

            # Numeric column stats — helps LLM answer avg/min/max queries
            numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                parts.append("Numeric column statistics:")
                for col in numeric_cols:
                    col_data = dataframe[col].dropna()
                    if len(col_data) > 0:
                        parts.append(
                            f"  {col} — values: {col_data.tolist()}, "
                            f"min: {col_data.min()}, max: {col_data.max()}, "
                            f"average: {round(col_data.mean(), 4)}, "
                            f"sum: {round(col_data.sum(), 4)}"
                        )

            # All rows in readable format
            parts.append("\nAll rows:")
            for _, row in dataframe.iterrows():
                row_text = ", ".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                    if pd.notna(val)
                )
                parts.append(f"  - {row_text}")

            return "\n".join(parts)

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Table summarization failed: {e}")
            raise ExtractionError(f"Failed to summarize table: {e}") from e

    def summarize_row(self, row: pd.Series, table_title: str = "") -> str:
        """
        Convert a single DataFrame row into a readable string.
        Includes table title prefix so the row chunk is self-contained.
        """
        if row is None or row.empty:
            raise InvalidInputError("Row is empty")
        try:
            values = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            row_text = ", ".join(values)
            if table_title:
                return f"[{table_title}] {row_text}"
            return row_text
        except Exception as e:
            logger.error(f"Row summarization failed: {e}")
            raise ExtractionError(f"Failed to summarize row: {e}") from e