# utils/document_loaders/loader_factory.py

import os
from typing import Type
from .base import BaseDocumentLoader
from .pdf_loader import PDFLoader
# from .pdf_loader import PDFOCRLoader  # <- uncomment when ready
from .docx_loader import DocxLoader
from .epub_loader import EpubLoader

# Map extension→loader class. Add more as needed (.txt, .md, etc.)
_LOADER_MAP: dict[str, Type[BaseDocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".doc": DocxLoader,   # if you handle .doc via conversion in DocxLoader
    ".epub": EpubLoader,
    # you could add ".txt": TxtLoader, etc.
    # you could add ".md": MarkdownLoader, etc.
}

def get_loader_for(path: str) -> BaseDocumentLoader:
    """
    Look at the file extension of `path` and return an instance of the appropriate loader.
    Raises ValueError if unsupported.
    """
    ext = os.path.splitext(path.lower())[1]

    # Fetch proper loader class
    LoaderCls = _LOADER_MAP.get(ext)

    # Graceful error catch
    if LoaderCls is None:
        raise ValueError(f"Unsupported document type '{ext}'. Available: {list(_LOADER_MAP.keys())}")
    return LoaderCls()
