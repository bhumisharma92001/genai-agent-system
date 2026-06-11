from pathlib import Path

import pdfplumber
import pandas as pd
from utils.logger import logger
from exceptions.custom_errors import ExtractionError, InvalidInputError

class TableExtractor:

    def __init__(self):
        self.handlers = {
            ".pdf": self._extract_pdf_tables,
            ".csv": self._extract_dataframe_tables,
            ".xlsx": self._extract_dataframe_tables,
            ".xls": self._extract_dataframe_tables
        }

    def extract(self,file_path: str) -> list[dict]:
        if not file_path:
            raise ValueError("file_path cannot be empty")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            extension = path.suffix.lower()
            handler = self.handlers.get(extension)
            if not handler:
                return []
            return handler(file_path)

        except Exception as e:
            raise RuntimeError("Failed to extract tables") from e

    def _extract_pdf_tables(self,file_path: str) -> list[dict]:
        extracted_tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages,start=1):
                    tables = (page.extract_tables())

                    for table in tables:
                        if not table:
                            continue

                        if len(table) < 2:
                            continue

                        df = pd.DataFrame(table[1:],columns=table[0])
                        extracted_tables.append({"dataframe": df,"page": page_number})

            return extracted_tables

        except Exception as e:
            raise RuntimeError("Failed to extract PDF tables") from e

    def _extract_dataframe_tables(self,file_path: str) -> list[dict]:

        try:
            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                dataframe = pd.read_csv(file_path)

            else:
                dataframe = pd.read_excel(file_path)

            return [{"dataframe": dataframe,"page": 1}]

        except Exception as e:
            logger.error(f"Dataframe extraction failed: {str(e)}")
            raise ExtractionError(f"Failed to extract dataframe tables: {e}") from e