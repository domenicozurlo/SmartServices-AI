# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None
    # When True the response includes context_groups with image metadata
    multimodal: Optional[bool] = False


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4


# ---------------------------------------------------------------------------
# Multimodal OCR ingest models
# ---------------------------------------------------------------------------

class StructuredOCRImageRecord(BaseModel):
    image_id: str
    url: str
    caption: str = ""
    image_summary: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str = "image/png"
    size_bytes: int = 0
    bbox: Optional[Dict[str, Any]] = None
    image_hash: Optional[str] = None


class StructuredOCRPage(BaseModel):
    page: int
    markdown: str
    images: List[StructuredOCRImageRecord] = []
    dimensions: Optional[Dict[str, Any]] = {}


class StructuredOCRResult(BaseModel):
    file_id: str
    source_file: str
    pages: List[StructuredOCRPage]


class StructuredEmbedRequest(BaseModel):
    file_id: str
    structured_ocr: StructuredOCRResult
    entity_id: Optional[str] = None
    # Chunking parameters (override defaults when provided)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    expansion_depth_before: Optional[int] = 1
    expansion_depth_after: Optional[int] = 1
    # Base URL of the original document in LibreChat storage (e.g. /images/uploads/...)
    # Used to build clickable per-page citation links: source_url_base#page=N
    source_url_base: Optional[str] = None
