"""
Advanced RAG Engine for AI Teacher

Responsibilities:
- Load PDF, TXT, DOCX, DOC and PPTX learning materials
- Preserve source metadata such as page/slide number
- Split material into retrieval-friendly chunks
- Create embeddings
- Store/retrieve chunks using Chroma
- Return grounded context with source references
- Support topic-based retrieval
- Avoid requiring Gemini/API keys for local RAG tests
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

COLLECTION_NAME = "educational_material"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# PPTX loader
# ---------------------------------------------------------------------------

class PPTXLoader:
    """
    Lightweight PowerPoint loader.

    Extracts:
    - slide title
    - text boxes
    - bullet points
    - table text
    - notes where available

    Each slide becomes a LangChain Document with slide metadata.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> List[Document]:
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError(
                "python-pptx is required for PPTX support. "
                "Install it with: pip install python-pptx"
            ) from exc

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"PowerPoint file not found: {self.file_path}"
            )

        presentation = Presentation(str(self.file_path))

        documents: List[Document] = []

        for slide_number, slide in enumerate(
            presentation.slides,
            start=1,
        ):
            parts: List[str] = []

            # ---------------------------------------------------------------
            # Slide title
            # ---------------------------------------------------------------

            title = ""

            try:
                if slide.shapes.title is not None:
                    title = slide.shapes.title.text.strip()
            except Exception:
                title = ""

            if title:
                parts.append(f"Slide Title: {title}")

            # ---------------------------------------------------------------
            # Shapes / text boxes / tables
            # ---------------------------------------------------------------

            for shape in slide.shapes:

                # Skip the title because it has already been captured.
                if getattr(shape, "is_placeholder", False):
                    try:
                        if shape == slide.shapes.title:
                            continue
                    except Exception:
                        pass

                # Normal text shape
                if hasattr(shape, "text"):
                    text = str(shape.text).strip()

                    if text:
                        parts.append(text)

                # Tables
                if getattr(shape, "has_table", False):
                    try:
                        table = shape.table

                        for row in table.rows:
                            cells = []

                            for cell in row.cells:
                                value = cell.text.strip()

                                if value:
                                    cells.append(value)

                            if cells:
                                parts.append(" | ".join(cells))

                    except Exception:
                        pass

            # ---------------------------------------------------------------
            # Speaker notes
            # ---------------------------------------------------------------

            try:
                notes_slide = slide.notes_slide

                note_parts = []

                for shape in notes_slide.shapes:
                    if hasattr(shape, "text"):
                        note_text = str(shape.text).strip()

                        if note_text:
                            note_parts.append(note_text)

                if note_parts:
                    parts.append(
                        "Speaker Notes:\n"
                        + "\n".join(note_parts)
                    )

            except Exception:
                # Some PPTX files do not expose notes cleanly.
                pass

            slide_text = "\n\n".join(
                part.strip()
                for part in parts
                if part and part.strip()
            ).strip()

            if not slide_text:
                continue

            documents.append(
                Document(
                    page_content=slide_text,
                    metadata={
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "file_type": "pptx",
                        "slide": slide_number,
                        "page": None,
                        "location": f"Slide {slide_number}",
                    },
                )
            )

        return documents


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Advanced retrieval-augmented generation backend.

    Usage:

        rag = RAGEngine()

        rag.ingest_file("chapter.pdf")

        results = rag.retrieve("Ohm's law")

        context = rag.build_grounded_context(results)
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".docx",
        ".doc",
        ".pptx",
    }

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.persist_directory = Path(
            persist_directory or CHROMA_DIR
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embeddings = None
        self._vectorstore = None

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------

    def _get_embeddings(self):
        """
        Lazily initialize embeddings.

        This is important for application startup performance.
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

        return self._embeddings

    # -----------------------------------------------------------------------
    # Vector store
    # -----------------------------------------------------------------------

    def _get_vectorstore(self):
        """
        Lazily initialize Chroma.
        """

        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=str(
                    self.persist_directory
                ),
            )

        return self._vectorstore

    # -----------------------------------------------------------------------
    # File validation
    # -----------------------------------------------------------------------

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            return {
                "valid": False,
                "error": "File does not exist.",
                "file_type": path.suffix.lower(),
            }

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            return {
                "valid": False,
                "error": (
                    f"Unsupported file type: {extension}. "
                    f"Supported types: "
                    f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
                ),
                "file_type": extension,
            }

        if path.stat().st_size == 0:
            return {
                "valid": False,
                "error": "File is empty.",
                "file_type": extension,
            }

        return {
            "valid": True,
            "error": None,
            "file_type": extension,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }

    # -----------------------------------------------------------------------
    # Document loading
    # -----------------------------------------------------------------------

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a supported educational file.
        """

        validation = self.validate_file(file_path)

        if not validation["valid"]:
            raise ValueError(validation["error"])

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".pdf":
            documents = PyPDFLoader(
                str(path)
            ).load()

        elif extension == ".txt":
            documents = TextLoader(
                str(path),
                encoding="utf-8",
            ).load()

        elif extension in {".docx", ".doc"}:
            documents = Docx2txtLoader(
                str(path)
            ).load()

        elif extension == ".pptx":
            documents = PPTXLoader(
                str(path)
            ).load()

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return self._normalize_metadata(
            documents,
            path,
        )

    # -----------------------------------------------------------------------
    # Metadata normalization
    # -----------------------------------------------------------------------

    def _normalize_metadata(
        self,
        documents: Sequence[Document],
        source_path: Path,
    ) -> List[Document]:
        """
        Normalize metadata across PDF, DOCX, TXT and PPTX.
        """

        normalized = []

        extension = source_path.suffix.lower().lstrip(".")

        for index, document in enumerate(documents):
            metadata = dict(
                document.metadata or {}
            )

            metadata.setdefault(
                "source",
                str(source_path),
            )

            metadata.setdefault(
                "filename",
                source_path.name,
            )

            metadata.setdefault(
                "file_type",
                extension,
            )

            # PDF loaders generally provide page as zero-based.
            if extension == "pdf":
                if metadata.get("page") is not None:
                    try:
                        metadata["page"] = (
                            int(metadata["page"]) + 1
                        )
                    except Exception:
                        pass

                    metadata["location"] = (
                        f"Page {metadata['page']}"
                    )

            # PPTX loader already provides slide.
            if extension == "pptx":
                slide = metadata.get("slide")

                if slide is not None:
                    metadata["location"] = (
                        f"Slide {slide}"
                    )

            # Generic fallback.
            metadata.setdefault(
                "document_index",
                index,
            )

            normalized.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return normalized

    # -----------------------------------------------------------------------
    # Text cleaning
    # -----------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text while preserving meaningful structure.
        """

        if not text:
            return ""

        text = text.replace(
            "\x00",
            " ",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # -----------------------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------------------

    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> List[Document]:
        """
        Split documents while preserving metadata.
        """

        cleaned_documents = []

        for document in documents:
            cleaned = self.clean_text(
                document.page_content
            )

            if not cleaned:
                continue

            cleaned_documents.append(
                Document(
                    page_content=cleaned,
                    metadata=dict(
                        document.metadata or {}
                    ),
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
            ],
        )

        chunks = splitter.split_documents(
            cleaned_documents
        )

        # Add stable chunk metadata.
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index

        return chunks

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest_documents(
        self,
        documents: Sequence[Document],
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest already-loaded documents.
        """

        if clear_existing:
            self.clear_collection()

        chunks = self.split_documents(
            documents
        )

        if not chunks:
            return {
                "success": False,
                "documents": len(documents),
                "chunks": 0,
                "message": "No usable text was found.",
            }

        vectorstore = self._get_vectorstore()

        vectorstore.add_documents(
            chunks
        )

        return {
            "success": True,
            "documents": len(documents),
            "chunks": len(chunks),
            "message": (
                f"Indexed {len(chunks)} chunks "
                f"from {len(documents)} source sections."
            ),
        }

    def ingest_file(
        self,
        file_path: str,
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Load + chunk + index one educational file.
        """

        documents = self.load_file(
            file_path
        )

        result = self.ingest_documents(
            documents,
            clear_existing=clear_existing,
        )

        result["filename"] = Path(
            file_path
        ).name

        result["file_type"] = Path(
            file_path
        ).suffix.lower()

        return result

    def ingest_files(
        self,
        file_paths: Sequence[str],
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest multiple educational materials.
        """

        if clear_existing:
            self.clear_collection()

        total_documents = 0
        total_chunks = 0

        files_processed = []
        errors = []

        for file_path in file_paths:
            try:
                documents = self.load_file(
                    file_path
                )

                result = self.ingest_documents(
                    documents,
                    clear_existing=False,
                )

                total_documents += result.get(
                    "documents",
                    0,
                )

                total_chunks += result.get(
                    "chunks",
                    0,
                )

                files_processed.append(
                    Path(file_path).name
                )

            except Exception as exc:
                errors.append(
                    {
                        "file": str(file_path),
                        "error": str(exc),
                    }
                )

        return {
            "success": len(errors) == 0,
            "files_processed": files_processed,
            "documents": total_documents,
            "chunks": total_chunks,
            "errors": errors,
        }

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks.
        """

        query = (query or "").strip()

        if not query:
            return []

        vectorstore = self._get_vectorstore()

        top_k = max(
            1,
            int(top_k),
        )

        if filter_metadata:
            return vectorstore.similarity_search(
                query,
                k=top_k,
                filter=filter_metadata,
            )

        return vectorstore.similarity_search(
            query,
            k=top_k,
        )

    # -----------------------------------------------------------------------
    # Retrieval with scores
    # -----------------------------------------------------------------------

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks together with similarity scores.
        """

        query = (query or "").strip()

        if not query:
            return []

        vectorstore = self._get_vectorstore()

        results = vectorstore.similarity_search_with_score(
            query,
            k=max(
                1,
                int(top_k),
            ),
        )

        formatted = []

        for document, score in results:
            formatted.append(
                {
                    "document": document,
                    "score": float(score),
                    "content": document.page_content,
                    "metadata": dict(
                        document.metadata or {}
                    ),
                }
            )

        return formatted

    # -----------------------------------------------------------------------
    # Grounded context
    # -----------------------------------------------------------------------

    def build_grounded_context(
        self,
        documents: Sequence[Document],
        include_source: bool = True,
    ) -> str:
        """
        Convert retrieved documents into structured context for Gemini.

        The model receives explicit source boundaries so it can distinguish
        retrieved evidence from its own general knowledge.
        """

        if not documents:
            return (
                "NO_RELEVANT_SOURCE_MATERIAL_FOUND.\n"
                "Do not claim that the uploaded material states "
                "information that was not retrieved."
            )

        sections = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata or {}

            filename = metadata.get(
                "filename",
                "Unknown source",
            )

            location = metadata.get(
                "location",
                "",
            )

            source_line = filename

            if location:
                source_line += (
                    f" — {location}"
                )

            if include_source:
                header = (
                    f"[SOURCE {index}: {source_line}]"
                )
            else:
                header = (
                    f"[SOURCE {index}]"
                )

            sections.append(
                "\n".join(
                    [
                        header,
                        document.page_content.strip(),
                        f"[END SOURCE {index}]",
                    ]
                )
            )

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------
    # Source references
    # -----------------------------------------------------------------------

    def get_source_references(
        self,
        documents: Sequence[Document],
    ) -> List[Dict[str, Any]]:
        """
        Return clean source references for UI display.
        """

        references = []

        seen = set()

        for document in documents:
            metadata = document.metadata or {}

            filename = metadata.get(
                "filename",
                "Unknown source",
            )

            location = metadata.get(
                "location",
                "",
            )

            key = (
                filename,
                location,
            )

            if key in seen:
                continue

            seen.add(key)

            references.append(
                {
                    "filename": filename,
                    "location": location,
                    "file_type": metadata.get(
                        "file_type",
                        "",
                    ),
                    "source": metadata.get(
                        "source",
                        "",
                    ),
                }
            )

        return references

    # -----------------------------------------------------------------------
    # Grounded retrieval helper
    # -----------------------------------------------------------------------

    def retrieve_grounded(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        One-call retrieval helper.

        Returns:
        - retrieved documents
        - grounded context
        - source references
        """

        documents = self.retrieve(
            query,
            top_k=top_k,
        )

        return {
            "query": query,
            "documents": documents,
            "context": self.build_grounded_context(
                documents
            ),
            "sources": self.get_source_references(
                documents
            ),
            "count": len(documents),
        }

    # -----------------------------------------------------------------------
    # Collection information
    # -----------------------------------------------------------------------

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Return information about the current Chroma collection.
        """

        try:
            collection = self._get_vectorstore()

            count = collection._collection.count()

            return {
                "collection": self.collection_name,
                "chunks": count,
                "persist_directory": str(
                    self.persist_directory
                ),
            }

        except Exception as exc:
            return {
                "collection": self.collection_name,
                "chunks": 0,
                "persist_directory": str(
                    self.persist_directory
                ),
                "error": str(exc),
            }

    # -----------------------------------------------------------------------
    # Clear collection
    # -----------------------------------------------------------------------

    def clear_collection(self) -> bool:
        """
        Clear the current Chroma collection.
        """

        self._vectorstore = None

        try:
            from chromadb import PersistentClient

            client = PersistentClient(
                path=str(
                    self.persist_directory
                )
            )

            try:
                client.delete_collection(
                    self.collection_name
                )
            except Exception:
                pass

        except Exception:
            # Fallback: remove the local Chroma directory.
            if self.persist_directory.exists():
                try:
                    shutil.rmtree(
                        self.persist_directory
                    )
                except Exception:
                    pass

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._vectorstore = None

        return True

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Check whether the RAG backend can initialize.
        """

        try:
            self._get_embeddings()

            self._get_vectorstore()

            info = self.get_collection_info()

            return {
                "healthy": True,
                "embedding_model": (
                    self.embedding_model_name
                ),
                "collection": self.collection_name,
                "chunks": info.get(
                    "chunks",
                    0,
                ),
            }

        except Exception as exc:
            return {
                "healthy": False,
                "embedding_model": (
                    self.embedding_model_name
                ),
                "collection": self.collection_name,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Backward-compatible helper functions
# ---------------------------------------------------------------------------

def create_rag_engine(
    persist_directory: Optional[str] = None,
) -> RAGEngine:
    """
    Convenience constructor.
    """

    return RAGEngine(
        persist_directory=persist_directory
    )


def load_and_index(
    file_path: str,
    persist_directory: Optional[str] = None,
    clear_existing: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for loading and indexing one file.
    """

    engine = create_rag_engine(
        persist_directory
    )

    return engine.ingest_file(
        file_path,
        clear_existing=clear_existing,
    )


def search_material(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for retrieval.
    """

    engine = create_rag_engine(
        persist_directory
    )

    return engine.retrieve_grounded(
        query,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Local tests
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 70)
    print("ADVANCED RAG ENGINE TESTS")
    print("=" * 70)

    engine = RAGEngine(
        persist_directory=str(
            DATA_DIR / "rag_test_db"
        )
    )

    # ---------------------------------------------------------------
    # Test 1: initialization
    # ---------------------------------------------------------------

    print("\nTEST 1: Engine initialization")

    assert engine is not None
    assert engine.collection_name == COLLECTION_NAME

    print("PASS")

    # ---------------------------------------------------------------
    # Test 2: supported extensions
    # ---------------------------------------------------------------

    print("\nTEST 2: Supported file types")

    expected = {
        ".pdf",
        ".txt",
        ".docx",
        ".doc",
        ".pptx",
    }

    assert expected.issubset(
        engine.SUPPORTED_EXTENSIONS
    )

    print("PASS")

    # ---------------------------------------------------------------
    # Test 3: clean text
    # ---------------------------------------------------------------

    print("\nTEST 3: Text cleaning")

    cleaned = engine.clean_text(
        "Hello   world.\n\n\nThis is a test."
    )

    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned

    print("PASS")

    # ---------------------------------------------------------------
    # Test 4: document chunking
    # ---------------------------------------------------------------

    print("\nTEST 4: Document chunking")

    document = Document(
        page_content=(
            "Ohm's law states that voltage equals "
            "current multiplied by resistance. "
            * 30
        ),
        metadata={
            "filename": "test.txt",
            "file_type": "txt",
            "location": "Page 1",
        },
    )

    chunks = engine.split_documents(
        [document]
    )

    assert len(chunks) >= 1

    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert (
            chunk.metadata["filename"]
            == "test.txt"
        )

    print(
        f"PASS — generated {len(chunks)} chunks"
    )

    # ---------------------------------------------------------------
    # Test 5: grounded context
    # ---------------------------------------------------------------

    print("\nTEST 5: Grounded context generation")

    context = engine.build_grounded_context(
        chunks[:2]
    )

    assert "[SOURCE 1:" in context
    assert "[END SOURCE 1]" in context

    print("PASS")

    # ---------------------------------------------------------------
    # Test 6: source references
    # ---------------------------------------------------------------

    print("\nTEST 6: Source references")

    references = engine.get_source_references(
        chunks
    )

    assert len(references) >= 1
    assert (
        references[0]["filename"]
        == "test.txt"
    )

    print("PASS")

    # ---------------------------------------------------------------
    # Test 7: PPTX loader availability
    # ---------------------------------------------------------------

    print("\nTEST 7: PPTX loader")

    try:
        from pptx import Presentation

        assert Presentation is not None

        print("PASS — python-pptx available")

    except ImportError:
        print(
            "SKIPPED — install python-pptx "
            "to enable PPTX support"
        )

    # ---------------------------------------------------------------
    # Test 8: empty query
    # ---------------------------------------------------------------

    print("\nTEST 8: Empty retrieval query")

    result = engine.retrieve(
        ""
    )

    assert result == []

    print("PASS")

    # ---------------------------------------------------------------
    # Test 9: empty grounded context
    # ---------------------------------------------------------------

    print("\nTEST 9: Empty grounded context")

    empty_context = (
        engine.build_grounded_context([])
    )

    assert (
        "NO_RELEVANT_SOURCE_MATERIAL_FOUND"
        in empty_context
    )

    print("PASS")

    # ---------------------------------------------------------------
    # Test 10: file validation
    # ---------------------------------------------------------------

    print("\nTEST 10: File validation")

    validation = engine.validate_file(
        "nonexistent_file.pdf"
    )

    assert validation["valid"] is False

    print("PASS")

    print("\n" + "=" * 70)
    print("ALL LOCAL RAG TESTS PASSED")
    print("=" * 70)
    print(
        "\nNote: embedding/vector database initialization "
        "is intentionally not performed by the local tests."
    )


if __name__ == "__main__":
    run_tests()