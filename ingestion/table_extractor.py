from pathlib import Path

import pdfplumber
import pandas as pd


class TableExtractor:

    def extract(
        self,
        file_path: str
    ) -> list[dict]:

        if not file_path:

            raise ValueError(
                "file_path cannot be empty"
            )

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        try:

            if file_path.endswith(".pdf"):

                return self._extract_pdf_tables(
                    file_path
                )

            if (
                file_path.endswith(".csv")
                or file_path.endswith(".xlsx")
                or file_path.endswith(".xls")
            ):

                return (
                    self._extract_dataframe_tables(
                        file_path
                    )
                )

            return []

        except Exception as e:

            raise RuntimeError(
                "Failed to extract tables"
            ) from e

    def _extract_pdf_tables(
        self,
        file_path: str
    ) -> list[dict]:

        extracted_tables = []

        try:

            with pdfplumber.open(
                file_path
            ) as pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    tables = (
                        page.extract_tables()
                    )

                    for table in tables:

                        if not table:

                            continue

                        if len(table) < 2:

                            continue

                        df = pd.DataFrame(
                            table[1:],
                            columns=table[0]
                        )

                        extracted_tables.append(
                            {
                                "dataframe": df,
                                "page": page_number
                            }
                        )

            return extracted_tables

        except Exception as e:

            raise RuntimeError(
                "Failed to extract PDF tables"
            ) from e

    def _extract_dataframe_tables(
        self,
        file_path: str
    ) -> list[dict]:

        try:

            if file_path.endswith(".csv"):

                dataframe = pd.read_csv(
                    file_path
                )

            else:

                dataframe = pd.read_excel(
                    file_path
                )

            return [
                {
                    "dataframe": dataframe,
                    "page": 1
                }
            ]

        except Exception as e:

            raise RuntimeError(
                "Failed to extract dataframe tables"
            ) from e