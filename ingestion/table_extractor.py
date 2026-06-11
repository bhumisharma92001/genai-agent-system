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
            ".xls": self._extract_dataframe_tables,
        }

    def extract(self, file_path: str) -> list[dict]:
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
        except (ValueError, FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to extract tables: {e}") from e

    def _extract_pdf_tables(self, file_path: str) -> list[dict]:
        extracted_tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                table_index = 0
                for page_number, page in enumerate(pdf.pages, start=1):
                    # Try to find a nearby heading for this table (text just above)
                    page_text_lines = (page.extract_text() or "").splitlines()
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        raw_headers = table[0]

                        # Clean headers — replace None/empty with "Col_N"
                        headers = [
                            str(h).strip() if h and str(h).strip() else f"Col_{i}"
                            for i, h in enumerate(raw_headers)
                        ]

                        df = pd.DataFrame(table[1:], columns=headers)

                        # Drop fully empty rows
                        df = df.dropna(how="all").reset_index(drop=True)
                        if df.empty:
                            continue

                        # Attempt to infer a table title from page text
                        # Look for the last non-empty line before typical table keywords
                        table_title = _infer_title(page_text_lines, table_index, page_number)

                        extracted_tables.append({
                            "dataframe": df,
                            "page": page_number,
                            "table_title": table_title,
                            "column_headers": headers,
                        })
                        table_index += 1

            return extracted_tables

        except ExtractionError:
            raise
        except Exception as e:
            logger.error(f"PDF table extraction failed: {e}")
            raise ExtractionError(f"Failed to extract PDF tables: {e}") from e

    def _extract_dataframe_tables(self, file_path: str) -> list[dict]:
        try:
            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            df = df.dropna(how="all").reset_index(drop=True)
            headers = [str(c).strip() for c in df.columns.tolist()]
            df.columns = headers

            return [{
                "dataframe": df,
                "page": 1,
                "table_title": Path(file_path).stem,   # filename as title
                "column_headers": headers,
            }]

        except Exception as e:
            logger.error(f"Dataframe extraction failed: {e}")
            raise ExtractionError(f"Failed to extract dataframe tables: {e}") from e


def _infer_title(page_lines: list[str], table_index: int, page_number: int) -> str:
    """
    Best-effort: scan page lines for a short heading-like line.
    Falls back to 'Table <N> (Page <P>)' if nothing useful found.
    """
    candidates = [
        line.strip() for line in page_lines
        if line.strip() and len(line.strip()) < 80 and not line.strip().startswith("|")
    ]
    # Last short line on page is often the table caption/heading
    if candidates:
        return candidates[-1]
    return f"Table {table_index + 1} (Page {page_number})"