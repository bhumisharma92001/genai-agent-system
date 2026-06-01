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
                dataframe.head(5)
                .to_dict(
                    orient="records"
                )
            )

            summary = (
                f"Table contains "
                f"{row_count} rows. "
                f"Columns: "
                f"{', '.join(columns)}. "
            )

            # Add unique values for small-cardinality columns
            for column in dataframe.columns:

                unique_values = (
                    dataframe[column]
                    .dropna()
                    .unique()
                )

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

        except Exception as e:

            raise RuntimeError(
                "Failed to summarize table"
            ) from e

    def summarize_row(
        self,
        row: pd.Series
    ) -> str:

        values = []

        for column, value in row.items():

            values.append(
                f"{column}: {value}"
            )

        return ", ".join(values)