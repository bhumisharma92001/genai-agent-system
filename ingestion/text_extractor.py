# ingestion/text_extractor.py

import pdfplumber
from docx import Document
import pandas as pd


class TextExtractor:

    def extract(
        self,
        file_path: str
    ) -> str:

        if file_path.endswith(".pdf"):
            return self._extract_pdf_text(file_path)

        if file_path.endswith(".docx"):
            return self._extract_docx_text(file_path)

        if (
            file_path.endswith(".csv")
            or file_path.endswith(".xlsx")
            or file_path.endswith(".xls")
        ):
            return self._extract_tabular_text(file_path)

        raise ValueError("Unsupported file type")

    @staticmethod
    def _extract_pdf_text(
        file_path: str
    ) -> str:

        texts = []

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if not page_text:
                    continue

                lines = page_text.split("\n")

                cleaned_lines = []

                skip_table = False

                for line in lines:

                    normalized = " ".join(
                        line.split()
                    )

                    # detect table start
                    if (
                        "Quarter Revenue Growth %" in normalized
                    ):
                        skip_table = True
                        continue

                    # stop skipping when next section starts
                    if (
                        skip_table
                        and normalized.startswith("3.")
                    ):
                        skip_table = False

                    if not skip_table:
                        cleaned_lines.append(line)

                cleaned_text = "\n".join(
                    cleaned_lines
                )

                texts.append(cleaned_text)

        return "\n".join(texts)

    @staticmethod
    def _extract_docx_text(
        file_path: str
    ) -> str:

        document = Document(file_path)

        paragraphs = []

        for para in document.paragraphs:

            text = para.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    @staticmethod
    def _extract_tabular_text(
        file_path: str
    ) -> str:

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        return df.to_string(index=False)