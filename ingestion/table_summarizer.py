import pandas as pd


class TableSummarizer:

    def summarize(
        self,
        dataframe: pd.DataFrame
    ) -> str:

        if dataframe.empty:

            raise ValueError(
                "Dataframe is empty"
            )

        try:

            columns = list(
                dataframe.columns
            )

            row_count = len(
                dataframe
            )

            preview_rows = (
                dataframe.head(3)
                .to_dict(
                    orient="records"
                )
            )

            summary = (
                f"Table contains "
                f"{row_count} rows "
                f"and columns: "
                f"{', '.join(columns)}. "
                f"Sample rows: "
                f"{preview_rows}"
            )

            return summary

        except Exception as e:

            raise RuntimeError(
                "Failed to summarize table"
            ) from e