# utils/document_loaders/docx_loader.py

from typing import List, Dict
from .base import BaseDocumentLoader
from langchain.schema import Document  # or you can define your own simple Document
import docx  # pip install python‐docx

class DocxLoader(BaseDocumentLoader):
    def load_documents(self, path: str) -> List[Document]:
        """
        Reads a .docx file via python‐docx. Splits each paragraph into a Document.
        If you also need classic .doc (old format), you could call `textract` or `pypandoc`
        to convert it to text first, then wrap in Documents.
        """
        doc = docx.Document(path)
        docs: List[Document] = []
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": None, "paragraph_index": i}
                )
            )
        return docs


import subprocess

def convert_doc_to_docx(doc_path: str) -> str:
    # e.g. "libreoffice --headless --convert-to docx <path>"
    output = subprocess.check_output([
        "libreoffice", "--headless", "--convert-to", "docx", doc_path
    ])
    return doc_path.replace(".doc", ".docx")