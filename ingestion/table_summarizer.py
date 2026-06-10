import pandas as pd
from utils.logger import logger
from exceptions.custom_errors import ExtractionError, InvalidInputError

class TableSummarizer:

    def summarize(self,dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            raise InvalidInputError("Dataframe is empty")

        try:
            columns = list(dataframe.columns)
            row_count = len(dataframe)
            preview_rows = (dataframe.head(5).to_dict(orient="records"))

            summary = (
                f"Table contains "
                f"{row_count} rows. "
                f"Columns: "
                f"{', '.join(columns)}. "
            )
            for column in dataframe.columns:
                unique_values = (dataframe[column].dropna().unique())

                if len(unique_values) <= 20:
                    summary += (
                        f"{column} values: "
                        f"{list(unique_values)}. "
                    )

            summary += (
                f"Sample rows: "
                f"{preview_rows}"
            )

            return summary

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Table summarization failed: {str(e)}")
            raise ExtractionError(f"Failed to summarize table: {e}") from e

    def summarize_row(self,row: pd.Series) -> str:
        if row is None or row.empty:
            raise InvalidInputError("Row is empty")
        try:
            values = []
            for column, value in row.items():
                values.append(f"{column}: {value}")
            return ", ".join(values)
        except Exception as e:
            logger.error(f"Row summarization failed: {str(e)}")
            raise ExtractionError(f"Failed to summarize row: {e}") from e