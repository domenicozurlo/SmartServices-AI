# app/utils/figure_extractor.py
"""
PDF figure extraction using pymupdf (fitz).

For each page in a PDF:
- Extracts embedded raster images (PNG/JPEG/etc.) skipping tiny decorative ones.
- Saves each image as PNG under {RAG_UPLOAD_DIR}/images/{file_id}/.
- Optionally generates an `image_summary` via OpenAI vision (describer.py).
- Returns page text with <image_id>…</image_id> placeholders injected after the
  page text so that chunk_page() can co-locate them with their surrounding text.

Public API
----------
    images_by_id, pages = await extract_figures_from_pdf(
        pdf_bytes, file_id, source_filename, describe=True
    )

    images_by_id : Dict[str, ImageRecord]
        image_id → {image_id, url, page, width, height, image_summary, source_file}

    pages : List[Tuple[int, str]]
        (page_num, page_text_with_placeholders)  — 0-indexed page numbers
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import RAG_UPLOAD_DIR, logger

# Guard so the module loads even without pymupdf installed.
try:
    import fitz  # pymupdf
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("figure_extractor: pymupdf not installed — PDF figure extraction disabled")

_IMAGES_SUBDIR = "images"
_MIN_IMAGE_BYTES = 2048    # skip images smaller than 2 KB (icons, decorations)
_MIN_DIMENSION = 40        # skip images narrower or shorter than 40 px


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_dir(file_id: str) -> Path:
    p = Path(RAG_UPLOAD_DIR) / _IMAGES_SUBDIR / file_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _image_url(file_id: str, image_id: str) -> str:
    return f"/content/image/{file_id}/{image_id}"


def _safe_img_name(raw_name: str, xref: int) -> str:
    name = (raw_name or f"img-{xref}").replace("/", "_").replace("\\", "_")
    # Remove extension so we can always save as .png
    return Path(name).stem


def _to_png_bytes(doc: "fitz.Document", xref: int) -> Optional[bytes]:
    """Render an image xref as PNG bytes. Returns None on failure."""
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:          # CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png = pix.tobytes("png")
        pix = None                          # free memory
        return png
    except Exception as exc:
        logger.warning("figure_extractor: cannot render xref=%d as PNG: %s", xref, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_figures_from_pdf(
    pdf_bytes: bytes,
    file_id: str,
    source_filename: str,
    describe: bool = True,
) -> Tuple[Dict[str, Any], List[Tuple[int, str]]]:
    """
    Extract figures from a PDF and build per-page text with image placeholders.

    Parameters
    ----------
    pdf_bytes       Raw bytes of the PDF file.
    file_id         Unique identifier for the source document (used for paths/ids).
    source_filename Original filename (e.g. "manuale.pdf") stored in metadata.
    describe        Whether to call OpenAI vision to generate image_summary.
                    Set to False to skip description (e.g. in tests or when
                    OPENAI_API_KEY is absent — describer.py already degrades
                    gracefully, so True is safe even without the key).

    Returns
    -------
    images_by_id    Dict mapping image_id → image metadata record.
    pages           List of (page_num, page_text_with_placeholders).
    """
    if not _FITZ_AVAILABLE:
        logger.warning("figure_extractor: pymupdf unavailable, returning plain text only")
        return {}, []

    from app.utils.describer import describe_image as _describe_image

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    img_dir = _image_dir(file_id)
    images_by_id: Dict[str, Any] = {}
    pages: List[Tuple[int, str]] = []

    for page_num, page in enumerate(doc):
        page_text: str = page.get_text("text") or ""
        image_list = page.get_images(full=True)
        placeholders: List[str] = []

        for img_info in image_list:
            # img_info: (xref, smask, width, height, bpc, colorspace, ...)
            xref: int = img_info[0]
            width: int = img_info[2]
            height: int = img_info[3]
            raw_name: str = img_info[7] or ""

            if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
                continue

            try:
                img_data = doc.extract_image(xref)
            except Exception as exc:
                logger.warning(
                    "figure_extractor: cannot extract xref=%d page=%d: %s", xref, page_num, exc
                )
                continue

            native_bytes = img_data.get("image", b"")
            if len(native_bytes) < _MIN_IMAGE_BYTES:
                continue

            stem = _safe_img_name(raw_name, xref)
            image_id = f"{file_id}_p{page_num}_img{stem}"

            # Skip if already extracted (same xref referenced from multiple pages)
            if image_id in images_by_id:
                placeholders.append(f"<image_id>{image_id}</image_id>")
                continue

            png_bytes = _to_png_bytes(doc, xref) or native_bytes
            out_path = img_dir / f"{image_id}.png"
            with open(out_path, "wb") as fh:
                fh.write(png_bytes)

            image_summary: Optional[str] = None
            if describe:
                try:
                    image_summary = await _describe_image(png_bytes)
                except Exception as exc:
                    logger.warning(
                        "figure_extractor: description failed for %s: %s", image_id, exc
                    )

            images_by_id[image_id] = {
                "image_id": image_id,
                "url": _image_url(file_id, image_id),
                "page": page_num,
                "width": width,
                "height": height,
                "image_summary": image_summary,
                "source_file": source_filename,
            }
            placeholders.append(f"<image_id>{image_id}</image_id>")

        # Append all placeholders for this page after its text block.
        # chunk_page() will then place them in the chunk closest to end-of-page
        # and preserve them atomically (see multimodal.py figure-safe flush).
        if placeholders:
            page_text = page_text.rstrip("\n") + "\n\n" + "\n".join(placeholders)

        pages.append((page_num, page_text))

    doc.close()
    return images_by_id, pages
