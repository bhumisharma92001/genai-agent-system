import pdfplumber
from docx import Document
from pathlib import Path

class TextExtractor:

    def extract(
        self,
        file_path: str
    ) -> str:

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
                return self._extract_pdf_text(file_path)

            if file_path.endswith(".docx"):
                return self._extract_docx_text(file_path)

            raise ValueError(
                "Unsupported text document type"
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to extract text"
            ) from e


    def _extract_pdf_text(
        self,
        file_path: str
        ) -> str:

        extracted_texts = []

        try:

            with pdfplumber.open(file_path) as pdf:

                for page in pdf.pages:

                    page_text = page.extract_text()

                    if page_text:
                        extracted_texts.append(page_text)

            return "\n".join(extracted_texts)

        except Exception as e:

            raise RuntimeError(
                "Failed to extract PDF text"
            ) from e


    def _extract_docx_text(
        self,
        file_path: str
    ) -> str:
        extracted_paragraphs = []
        try:
            document = Document(file_path)

            for para in document.paragraphs:

                text = para.text.strip()

                if text:

                    extracted_paragraphs.append(text)

            return "\n".join(extracted_paragraphs)

        except Exception as e:
            raise RuntimeError(
                "Failed to extract DOCX text"
            ) from e

