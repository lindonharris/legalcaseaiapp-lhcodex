# utils/document_loaders/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
from typing import List
from .base import BaseDocumentLoader

class PDFLoader(BaseDocumentLoader):
    def load_documents(self, path: str) -> List:
        """
        Use PyPDFLoader to read a PDF from `path`.
        Returns a list of langchain.schema.Document (with page_content + metadata.page).
        """
        loader = PyPDFLoader(path)
        return loader.load()
