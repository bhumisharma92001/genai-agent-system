import pdfplumber
from docx import Document
from pathlib import Path
import pandas as pd

class TextExtractor:
    def __init__(self):
        self.handlers = {
            ".pdf": self._extract_pdf_text,
            ".docx": self._extract_docx_text,
            ".csv": self._extract_csv_text,
            ".xlsx": self._extract_excel_text,
            ".xls": self._extract_excel_text
        }

    def extract(self,file_path: str) -> str:
        if not file_path:
            raise ValueError("file_path cannot be empty")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            extension = path.suffix.lower()
            handler = self.handlers.get(extension)
            if not handler:
                raise ValueError("Unsupported text document type")
            return handler(file_path)

        except Exception as e:
            raise RuntimeError("Failed to extract text") from e


    def _extract_pdf_text(self,file_path: str) -> str:
        extracted_texts = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_texts.append(page_text)
            return "\n".join(extracted_texts)

        except Exception as e:
            raise RuntimeError("Failed to extract PDF text") from e

    def _extract_docx_text(self,file_path: str) -> str:
        extracted_paragraphs = []
        try:
            document = Document(file_path)
            for para in document.paragraphs:
                text = para.text.strip()
                if text:
                    extracted_paragraphs.append(text)
            return "\n".join(extracted_paragraphs)

        except Exception as e:
            raise RuntimeError("Failed to extract DOCX text") from e

    def _extract_csv_text(self,file_path: str) -> str:
        dataframe = pd.read_csv(file_path)
        return dataframe.to_string(index=False)

    def _extract_excel_text(self,file_path: str) -> str:
        dataframe = pd.read_excel(file_path)
        return dataframe.to_string(index=False)