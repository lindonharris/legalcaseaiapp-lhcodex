# utils/document_loaders/base.py

from typing import List, Any

class BaseDocumentLoader:
    """
    Every subclass must implement `load_documents(path: str) -> List[Document]`,
    where each Document is something your text‐splitter can consume (e.g. has `.page_content` + `.metadata`).
    You can choose `langchain.schema.Document` or roll your own simple dict with keys {"page_content", "metadata"}.
    """

    def load_documents(self, path: str) -> List[Any]:
        """
        Given a local file path, return a list of “documents” (could be pages or just one big chunk).
        Each item must have `.page_content: str` and `.metadata: dict` (if you want to preserve page numbers).
        """
        raise NotImplementedError("Subclasses must override load_documents().")
