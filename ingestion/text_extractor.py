import pdfplumber
from docx import Document
from pathlib import Path
from utils.logger import logger
from exceptions.custom_errors import ExtractionError, InvalidInputError, UnsupportedFormatError

class TextExtractor:
    def __init__(self):
        self.handlers = {
            ".pdf": self._extract_pdf_text,
            ".docx": self._extract_docx_text,
        }

    def extract(self,file_path: str) -> list[dict]:
        if not file_path:
            raise InvalidInputError("file_path cannot be empty")
        path = Path(file_path)
        if not path.exists():
            raise InvalidInputError(f"File not found: {file_path}")
        try:
            extension = path.suffix.lower()
            handler = self.handlers.get(extension)
            if not handler:
                raise UnsupportedFormatError(f"Unsupported file type: {extension}")
            return handler(file_path)

        except (InvalidInputError, UnsupportedFormatError):
            raise
        except Exception as e:
            logger.error(f"Text extraction failed for: {file_path} — {str(e)}")
            raise ExtractionError(f"Failed to extract text: {e}") from e

    def _extract_pdf_text(self,file_path: str) -> list[dict]:
        extracted_texts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number,page in enumerate (pdf.pages,start=1):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_texts.append({"page_no": page_number, "text": page_text})
            return(extracted_texts)

        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            raise ExtractionError(f"Failed to extract PDF text: {e}") from e

    def _extract_docx_text(self,file_path: str) -> list[dict]:
        extracted_paragraphs = []
        try:
            document = Document(file_path)
            for para in document.paragraphs:
                text = para.text.strip()
                if text:
                    extracted_paragraphs.append(text)
            final_text="\n".join(extracted_paragraphs)
            return [{"page_no": 1,"text":final_text}]
        except Exception as e:
            logger.error(f"DOCX text extraction failed: {str(e)}")
            raise ExtractionError(f"Failed to extract DOCX text: {e}") from e