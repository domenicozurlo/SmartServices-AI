# app/routes/document_routes.py
import os
import uuid
import re
from pathlib import Path
import hashlib
import traceback
import aiofiles
import aiofiles.os
from shutil import copyfileobj
from typing import List, Iterable, Optional, Union, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from fastapi.responses import FileResponse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session
from functools import lru_cache
import asyncio

if TYPE_CHECKING:
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector
    from langchain_community.vectorstores.pgvector import PGVector as PgVector

from app.config import (
    logger,
    vector_store,
    VECTOR_DB_TYPE,
    VectorDBType,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_QUEUE_SIZE,
    RAG_DISTANCE_THRESHOLD,
)

# Warn once at import time if the user set a threshold under Atlas, where
# the score direction is inverted (Atlas vectorSearchScore: higher = better)
# and naive `score <= threshold` would keep the *weaker* matches. We scope
# the filter to pgvector only until we grow a first-class "min similarity"
# semantic for Atlas.
#
# Inspect the raw env var here rather than the parsed RAG_DISTANCE_THRESHOLD:
# the parser in app.config deliberately skips the float() cast under Atlas
# (so non-numeric stale values don't break startup), which means the parsed
# value is always None for Atlas — and relying on it would suppress the
# warning we want operators to see.
if (
    VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO
    and os.getenv("RAG_DISTANCE_THRESHOLD") not in (None, "")
):
    logger.warning(
        "RAG_DISTANCE_THRESHOLD is set but VECTOR_DB_TYPE=atlas-mongo; "
        "Atlas returns similarity scores (higher = better) which would "
        "invert the filter semantics, so the threshold will be ignored."
    )


def _apply_distance_threshold(documents):
    """Drop (doc, score) tuples whose distance exceeds RAG_DISTANCE_THRESHOLD.

    Only applied for pgvector, where similarity_search_with_score_by_vector
    returns a distance (lower = more similar). Skipped for Atlas because its
    score is a similarity (higher = better) and applying the same comparison
    would keep the weakest matches and drop the strongest.
    """
    if RAG_DISTANCE_THRESHOLD is None:
        return documents
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return documents
    return [(doc, score) for doc, score in documents if score <= RAG_DISTANCE_THRESHOLD]
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
    StructuredEmbedRequest,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
)
from app.utils.health import is_health_ok
from app.utils.describer import describe_domain_hints, get_domain_hints_status
from app.utils.multimodal import (
    apply_domain_hints,
    build_ingest_report,
    build_all_chunks,
    chunks_to_documents,
    detect_query_intent,
    expand_results,
    filter_ranked_results,
    merge_groups,
    normalize_search_query,
    rerank_results,
    groups_to_context_groups,
)

router = APIRouter()


def calculate_num_batches(total: int, batch_size: int) -> int:
    """Calculate the number of batches needed to process total items."""
    if batch_size <= 0:
        return 1
    return (total + batch_size - 1) // batch_size


def get_user_id(request: Request, entity_id: str = None) -> str:
    """Extract user ID from request or entity_id."""
    if not hasattr(request.state, "user"):
        return entity_id if entity_id else "public"
    else:
        return entity_id if entity_id else request.state.user.get("id")


async def save_upload_file_async(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def save_upload_file_sync(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file synchronously."""
    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(file.file, temp_file)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def validate_file_path(base_dir: str, file_path: str) -> Optional[str]:
    """Validate that file_path resolves within base_dir. Returns resolved absolute path or None."""
    if not file_path or not file_path.strip():
        return None
    try:
        allowed = Path(base_dir).resolve()
        requested = Path(os.path.join(base_dir, file_path)).resolve()
        requested.relative_to(allowed)
        return str(requested)
    except (ValueError, RuntimeError, TypeError, OSError):
        return None


def _make_unique_temp_path(user_id: str, filename: str) -> Optional[str]:
    """Build a unique temp file path under RAG_UPLOAD_DIR/{user_id}/ to prevent
    concurrent upload collisions. Returns a validated absolute path, or None if
    the raw filename would escape RAG_UPLOAD_DIR (path traversal rejection)."""
    # Validate the raw filename to reject traversal attempts
    if validate_file_path(RAG_UPLOAD_DIR, os.path.join(user_id, filename)) is None:
        return None
    # unique_name is stem + "_" + [0-9a-f]{32} + suffix — no path separators,
    # so it cannot escape the directory validated above.
    p = Path(filename)
    unique_name = f"{p.stem}_{uuid.uuid4().hex}{p.suffix}"
    return str(Path(RAG_UPLOAD_DIR, user_id, unique_name).resolve())


async def load_file_content(
    filename: str,
    content_type: str,
    file_path: str,
    executor,
    raw_text: bool = False,
) -> tuple:
    """Load file content using appropriate loader.

    Pass ``raw_text=True`` when the caller wants verbatim file contents (e.g.
    the ``/text`` endpoint) so text-formatted files are not semantically
    parsed.
    """
    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            filename, content_type, file_path, raw_text=raw_text
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, lambda: list(loader.lazy_load()))
        return data, known_type, file_ext
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


def extract_text_from_documents(documents: List[Document], file_ext: str) -> str:
    """Extract text content from loaded documents."""
    text_content = ""
    if documents:
        for doc in documents:
            if hasattr(doc, "page_content"):
                # Clean text if it's a PDF
                if file_ext == "pdf":
                    text_content += clean_text(doc.page_content) + "\n"
                else:
                    text_content += doc.page_content + "\n"

    # Remove trailing newline
    return text_content.rstrip("\n")


async def cleanup_temp_file_async(file_path: str) -> None:
    """Clean up temporary file asynchronously."""
    try:
        await aiofiles.os.remove(file_path)
    except Exception as e:
        logger.error(
            "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
            file_path,
            str(e),
            traceback.format_exc(),
        )


@router.get("/ids")
async def get_all_ids(request: Request):
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids(executor=request.app.state.thread_pool)
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(request: Request, ids: list[str] = Query(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure all requested ids exist
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(request: Request, document_ids: List[str] = Body(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                document_ids, executor=request.app.state.thread_pool
            )
            await vector_store.delete(
                ids=document_ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(document_ids)
            vector_store.delete(ids=document_ids)

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Cache the embedding function with LRU cache
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    authorized_documents = []

    try:
        search_query = normalize_search_query(body.query)
        embedding = get_cached_query_embedding(search_query)
        k_fetch = max(body.k * 3, body.k)

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k_fetch,
                filter={"file_id": {"$eq": body.file_id}},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=k_fetch, filter={"file_id": {"$eq": body.file_id}}
            )

        documents = rerank_results(search_query, _apply_distance_threshold(documents))
        documents = filter_ranked_results(documents, min_results=1)[: body.k]

        if not documents:
            return authorized_documents

        document, score = documents[0]
        doc_metadata = document.metadata
        doc_user_id = doc_metadata.get("user_id")

        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents = documents
        else:
            # If using entity_id and access denied, try again with user's actual ID
            if body.entity_id and hasattr(request.state, "user"):
                user_authorized = request.state.user.get("id")
                if doc_user_id == user_authorized:
                    authorized_documents = documents
                else:
                    if body.entity_id == doc_user_id:
                        logger.warning(
                            f"Entity ID {body.entity_id} matches document user_id but user {user_authorized} is not authorized"
                        )
                    else:
                        logger.warning(
                            f"Access denied for both entity ID {body.entity_id} and user {user_authorized} to document with user_id {doc_user_id}"
                        )
            else:
                logger.warning(
                    f"Unauthorized access attempt by user {user_authorized} to a document with user_id {doc_user_id}"
                )

        return authorized_documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _process_documents_async_pipeline(
    documents: List[Document],
    file_id: str,
    vector_store: "AsyncPgVector",
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents using async producer-consumer pattern for batched embedding and insertion.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: AsyncPgVector instance for document storage
        executor: ThreadPoolExecutor for concurrent operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    # Create queues for producer-consumer pattern
    # embedding_queue is bounded to limit document data held in memory.
    # results_queue is unbounded — it holds only small UUID lists, and the
    # drain loop runs after gather(), so bounding it would deadlock when
    # num_batches > maxsize.
    embedding_queue = asyncio.Queue(maxsize=EMBEDDING_MAX_QUEUE_SIZE)
    results_queue = asyncio.Queue()
    all_ids = []

    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Starting async pipeline for file %s: %d chunks with %d batch size",
        file_id,
        total_chunks,
        EMBEDDING_BATCH_SIZE,
    )

    async def batch_producer():
        """Produce document batches and put them in the queue."""
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * EMBEDDING_BATCH_SIZE
                end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
                batch_documents = documents[start_idx:end_idx]
                batch_ids = [file_id] * len(batch_documents)

                logger.info(
                    "Generating embeddings for batch %d/%d: chunks %d-%d",
                    batch_idx + 1,
                    num_batches,
                    start_idx,
                    end_idx - 1,
                )

                # Put batch in queue for processing
                await embedding_queue.put(
                    (batch_documents, batch_ids, batch_idx + 1, num_batches)
                )
        except Exception as e:
            logger.error("Error in batch producer: %s", e)
            raise
        finally:
            # Always signal end of production
            await embedding_queue.put(None)

    async def embedding_consumer():
        """Consume batches from queue, embed and insert into database."""
        try:
            while True:
                item = await embedding_queue.get()
                if item is None:  # End signal
                    embedding_queue.task_done()
                    break

                batch_documents, batch_ids, batch_num, total_batches = item

                logger.info(
                    "Inserting batch %d/%d into database (%d chunks)",
                    batch_num,
                    total_batches,
                    len(batch_documents),
                )

                try:
                    # Insert batch into database
                    batch_result_ids = await vector_store.aadd_documents(
                        batch_documents, ids=batch_ids, executor=executor
                    )
                    await results_queue.put(batch_result_ids)
                except Exception as e:
                    logger.error(
                        "Error processing batch %d/%d: %s", batch_num, total_batches, e
                    )
                    await results_queue.put(e)  # Put exception object
                finally:
                    embedding_queue.task_done()

        except Exception as e:
            logger.error("Fatal error in embedding consumer: %s", e)
            await results_queue.put(e)
            raise

    producer_task = None
    consumer_task = None

    try:
        # Start producer and consumer concurrently
        producer_task = asyncio.create_task(batch_producer())
        consumer_task = asyncio.create_task(embedding_consumer())

        # Wait for both to complete
        await asyncio.gather(producer_task, consumer_task, return_exceptions=False)

        # Collect results from all batches
        for _ in range(num_batches):
            result = await results_queue.get()
            if isinstance(result, Exception):
                raise result
            all_ids.extend(result)

        logger.info(
            "Async pipeline completed for file %s: %d embeddings created",
            file_id,
            len(all_ids),
        )

        return all_ids

    except Exception as e:
        logger.error("Pipeline failed for file %s: %s", file_id, e)
        if consumer_task is not None or producer_task is not None:
            # if one of the tasks is still running, cancel it
            if consumer_task is not None and not consumer_task.done():
                consumer_task.cancel()
            if producer_task is not None and not producer_task.done():
                producer_task.cancel()

            # Await cancelled tasks to ensure proper cleanup
            if consumer_task is None:
                await asyncio.gather(producer_task, return_exceptions=True)
            elif producer_task is None:
                await asyncio.gather(consumer_task, return_exceptions=True)
            else:
                await asyncio.gather(
                    consumer_task, producer_task, return_exceptions=True
                )

        # Attempt rollback only if we inserted something
        if all_ids:
            try:
                logger.warning("Performing rollback of file %s", file_id)
                await vector_store.delete(ids=[file_id], executor=executor)
                logger.info("Rollback completed for file %s", file_id)
            except Exception as cleanup_error:
                logger.error("Rollback failed for file %s: %s", file_id, cleanup_error)

        # Re-raise the original error
        raise


async def _process_documents_batched_sync(
    documents: List[Document],
    file_id: str,
    vector_store: Union["PgVector", "AtlasMongoVector"],
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents in batches using synchronous vector store operations.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: Synchronous vector store instance (ExtendedPgVector or AtlasMongoVector)
        executor: ThreadPoolExecutor for running sync operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    all_ids = []
    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Processing file %s with sync batching: %d batches of %d chunks each",
        file_id,
        num_batches,
        EMBEDDING_BATCH_SIZE,
    )

    loop = asyncio.get_running_loop()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * EMBEDDING_BATCH_SIZE
        end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
        batch_documents = documents[start_idx:end_idx]
        batch_ids = [file_id] * len(batch_documents)

        logger.info(
            "Processing batch %d/%d: chunks %d-%d (%d chunks)",
            batch_idx + 1,
            num_batches,
            start_idx,
            end_idx - 1,
            len(batch_documents),
        )

        try:
            # Wrap sync call in executor to avoid blocking the event loop
            batch_result_ids = await loop.run_in_executor(
                executor,
                lambda docs=batch_documents, ids=batch_ids: vector_store.add_documents(
                    docs, ids=ids
                ),
            )
            all_ids.extend(batch_result_ids)

        except Exception as batch_error:
            logger.error("Batch %d failed: %s", batch_idx + 1, batch_error)

            # Rollback entire file from vector store
            if (
                all_ids
            ):  # any batch succeeded (i.e., any chunks for this file were inserted)
                logger.warning("Rolling back file %s due to batch failure", file_id)
                try:
                    await loop.run_in_executor(
                        executor, lambda: vector_store.delete(ids=[file_id])
                    )
                    logger.info("Rollback completed for file %s", file_id)
                except Exception as rollback_error:
                    logger.error(
                        "Rollback failed for file %s: %s", file_id, rollback_error
                    )

            raise batch_error

    return all_ids


def generate_digest(page_content: str) -> str:
    return hashlib.md5(page_content.encode("utf-8", "ignore")).hexdigest()


def _prepare_documents_sync(
    data: Iterable[Document],
    file_id: str,
    user_id: str,
    clean_content: bool,
) -> List[Document]:
    """
    Synchronous document preparation - runs in executor to avoid blocking event loop.
    Handles text splitting, cleaning, and metadata preparation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    # If `clean_content` is True, clean the page_content of each document (remove null bytes)
    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    # Preparing documents with page content and metadata for insertion.
    return [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
    pre_chunked: bool = False,
) -> bool:
    # When documents are already chunked (multimodal path), skip the text splitter
    loop = asyncio.get_running_loop()
    if pre_chunked:
        # Still run through executor to stay non-blocking, but skip splitting
        docs = await loop.run_in_executor(
            executor,
            lambda: [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "file_id": file_id,
                        "user_id": user_id,
                        **{k: v for k, v in (doc.metadata or {}).items() if k not in ("user_id",)},
                    },
                )
                for doc in data
            ],
        )
    else:
        # Run document preparation in executor to avoid blocking the event loop
        docs = await loop.run_in_executor(
            executor,
            _prepare_documents_sync,
            data,
            file_id,
            user_id,
            clean_content,
        )

    try:
        if EMBEDDING_BATCH_SIZE <= 0:
            # synchronously embed the file and insert into vector store in one go
            if isinstance(vector_store, AsyncPgVector):
                ids = await vector_store.aadd_documents(
                    docs, ids=[file_id] * len(docs), executor=executor
                )
            else:
                ids = vector_store.add_documents(docs, ids=[file_id] * len(docs))
        else:
            # asynchronously embed the file and insert into vector store as it is embedding
            # to lessen memory impact and speed up slightly as the majority of the document
            # is inserted into db by the time it is fully embedded

            if isinstance(vector_store, AsyncPgVector):
                ids = await _process_documents_async_pipeline(
                    docs, file_id, vector_store, executor
                )
            else:
                # Fallback to batched processing for sync vector stores
                ids = await _process_documents_batched_sync(
                    docs, file_id, vector_store, executor
                )

        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        return {"message": "An error occurred while adding documents.", "error": str(e)}


async def cleanup_stale_vectors(file_id: str, ingest_run_id: str, executor=None) -> int:
    if not hasattr(vector_store, "_bind"):
        logger.warning(
            "Skipping stale vector cleanup for file_id=%s; vector store does not expose SQL bind",
            file_id,
        )
        return 0

    def _delete_stale() -> int:
        with Session(vector_store._bind) as session:
            result = session.execute(
                text(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'file_id' = :file_id
                      AND COALESCE(cmetadata->>'ingest_run_id', '') != :ingest_run_id
                    """
                ),
                {"file_id": file_id, "ingest_run_id": ingest_run_id},
            )
            session.commit()
            return int(result.rowcount or 0)

    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(executor, _delete_stale)
    logger.info(
        "Cleaned up %d stale vectors for file_id=%s ingest_run_id=%s",
        deleted,
        file_id,
        ingest_run_id,
    )
    return deleted


def _is_numbered_heading(doc: Document) -> bool:
    headings = doc.metadata.get("chunk_headings") or []
    if not isinstance(headings, list):
        return False
    return any(isinstance(heading, str) and re.match(r"^\s*\d+[\.)]\s+", heading) for heading in headings)


def _multimodal_expansion_depth(search_query: str, documents: list) -> tuple[int, int]:
    if detect_query_intent(search_query) != "how_to" or not documents:
        return 0, 0

    normalized = normalize_search_query(search_query)
    top_docs = [doc for doc, _ in documents[:2]]
    has_numbered_step = any(_is_numbered_heading(doc) for doc in top_docs)
    if not has_numbered_step:
        return 0, 0

    specific_actions = {
        "accendere",
        "caricare",
        "controllare",
        "reset",
        "resettare",
        "rimuovere",
        "spegnere",
    }
    if "usare" in normalized and not any(action in normalized for action in specific_actions):
        return 0, 3
    return 0, 1


async def load_neighbor_chunks(
    file_id: str,
    seed_docs: list,
    depth_before: int,
    depth_after: int,
    executor=None,
) -> dict:
    chunks_by_id = {
        doc.metadata["chunk_id"]: doc
        for doc in seed_docs
        if doc.metadata.get("chunk_id")
    }
    if not chunks_by_id or (depth_before <= 0 and depth_after <= 0):
        return chunks_by_id
    if not hasattr(vector_store, "_bind"):
        return chunks_by_id

    def _load(chunk_ids: list) -> list[Document]:
        if not chunk_ids:
            return []
        stmt = text(
            """
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'file_id' = :file_id
              AND cmetadata->>'chunk_id' IN :chunk_ids
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        with Session(vector_store._bind) as session:
            rows = session.execute(stmt, {"file_id": file_id, "chunk_ids": chunk_ids}).all()
            return [
                Document(page_content=row.document or "", metadata=row.cmetadata or {})
                for row in rows
            ]

    async def _load_missing(chunk_ids: list) -> list[Document]:
        missing = sorted({
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id and chunk_id not in chunks_by_id
        })
        if not missing:
            return []
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(executor, lambda: _load(missing))
        for doc in loaded:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunks_by_id[chunk_id] = doc
        return loaded

    backward_frontier = list(seed_docs)
    for _ in range(depth_before):
        loaded = await _load_missing([
            doc.metadata.get("previous_chunk_id")
            for doc in backward_frontier
        ])
        backward_frontier = loaded
        if not loaded:
            break

    forward_frontier = list(seed_docs)
    for _ in range(depth_after):
        loaded = await _load_missing([
            doc.metadata.get("next_chunk_id")
            for doc in forward_frontier
        ])
        forward_frontier = loaded
        if not loaded:
            break

    return chunks_by_id


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    file_path = validate_file_path(RAG_UPLOAD_DIR, document.filepath)

    # Check if the file exists and if it is within the allowed upload directory
    if file_path is None or not os.path.exists(file_path):
        logger.warning("Path validation failed for local embed: %s", document.filepath)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            document.filename, document.file_content_type, file_path
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            request.app.state.thread_pool, lambda: list(loader.lazy_load())
        )

        result = await store_data_in_vector_db(
            data,
            document.file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


@router.post("/embed")
async def embed_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None

    user_id = get_user_id(request, entity_id)
    validated_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_file_path is None:
        logger.warning("Path validation failed for embed: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_file_path,
            request.app.state.thread_pool,
        )

        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
        elif "error" in result:
            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )
    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error(
            "Error during file processing: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_file_path)

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
    }


@router.get("/documents/{id}/context")
async def load_document_context(request: Request, id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure the requested id exists
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)

    validated_temp_file_path = _make_unique_temp_path(user_id, uploaded_file.filename)

    if validated_temp_file_path is None:
        logger.warning(
            "Path validation failed for embed-upload: %s", uploaded_file.filename
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(uploaded_file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            uploaded_file.filename,
            uploaded_file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
        )

        result = await store_data_in_vector_db(
            data,
            file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during file processing | File: %s | Error: %s | Traceback: %s",
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    try:
        # Get the embedding of the query text
        search_query = normalize_search_query(body.query)
        embedding = get_cached_query_embedding(search_query)
        k_fetch = max(body.k * 3, body.k)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k_fetch,
                filter={"file_id": {"$in": body.file_ids}},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=k_fetch, filter={"file_id": {"$in": body.file_ids}}
            )

        documents = rerank_results(search_query, _apply_distance_threshold(documents))
        documents = filter_ranked_results(documents, min_results=1)[: body.k]

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def extract_text_from_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """
    Extract text content from an uploaded file without creating embeddings.
    Returns the raw text content for text parsing purposes.
    """
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_file_path is None:
        logger.warning("Path validation failed for text extraction: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
            raw_text=True,
        )

        # Extract text content from loaded documents
        text_content = extract_text_from_documents(data, file_ext)

        return {
            "text": text_content,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in extract_text_from_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during text extraction | File: %s | Error: %s | Traceback: %s",
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error during text extraction: {str(e)}",
            )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)


# ---------------------------------------------------------------------------
# Multimodal endpoints
# ---------------------------------------------------------------------------


@router.post("/embed-structured")
async def embed_structured_ocr(body: StructuredEmbedRequest, request: Request):
    """
    Ingest structured multimodal OCR data (pages + images) produced by LibreChat's OCR layer.

    Creates multimodal chunks that preserve image placeholders in `text`, use enriched
    `embedding_text` for vector search, and link chunks via previous/next references.
    Image records are stored inline in chunk metadata so retrieval can surface them.
    """
    if not hasattr(request.state, "user"):
        user_id = body.entity_id if body.entity_id else "public"
    else:
        user_id = body.entity_id if body.entity_id else request.state.user.get("id")

    file_id = body.file_id
    ocr = body.structured_ocr

    logger.info(
        "[multimodal] embed-structured file_id=%s pages=%d user=%s",
        file_id,
        len(ocr.pages),
        user_id,
    )

    # Build images_by_id lookup from all pages
    images_by_id: dict = {}
    for page in ocr.pages:
        for img in page.images:
            images_by_id[img.image_id] = img

    logger.info(
        "[multimodal] embed-structured: images_by_id size=%d, keys=%s",
        len(images_by_id),
        list(images_by_id.keys())[:10],
    )
    if images_by_id:
        sample = next(iter(images_by_id.values()))
        logger.info(
            "[multimodal] embed-structured: sample image: id=%s url=%s has_url=%s",
            getattr(sample, "image_id", "?"),
            getattr(sample, "url", "?")[:80],
            bool(getattr(sample, "url", "")),
        )

    # Determine chunk parameters
    chunk_size = body.chunk_size or int(
        __import__("os").environ.get("MULTIMODAL_CHUNK_SIZE", "800")
    )
    chunk_overlap = body.chunk_overlap or int(
        __import__("os").environ.get("MULTIMODAL_CHUNK_OVERLAP", "120")
    )

    loop = asyncio.get_running_loop()

    try:
        # Build chunks in executor to avoid blocking event loop
        chunks = await loop.run_in_executor(
            request.app.state.thread_pool,
            lambda: build_all_chunks(
                pages=ocr.pages,
                file_id=file_id,
                source_file=ocr.source_file,
                images_by_id=images_by_id,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            ),
        )

        logger.info(
            "[multimodal] embed-structured: %d chunks built for file_id=%s", len(chunks), file_id
        )

        domain_hints = await describe_domain_hints(
            source_file=ocr.source_file,
            chunk_texts=[chunk.get("text", "") for chunk in chunks],
        )
        if domain_hints:
            apply_domain_hints(chunks, domain_hints)
            logger.info(
                "[multimodal] embed-structured: domain hints generated for file_id=%s domain=%s terms=%d",
                file_id,
                domain_hints.get("domain", ""),
                len(domain_hints.get("key_terms", [])),
            )

        if chunks:
            sample_chunk = chunks[0]
            logger.info(
                "[multimodal] embed-structured: sample chunk: chunk_id=%s image_ids=%s images_count=%d",
                sample_chunk.get("chunk_id"),
                sample_chunk.get("image_ids"),
                len([
                    iid for iid in sample_chunk.get("image_ids", [])
                    if iid in images_by_id
                ]),
            )

        ingest_run_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk["ingest_run_id"] = ingest_run_id

        ingest_report = build_ingest_report(
            chunks,
            images_by_id,
            domain_hints,
            get_domain_hints_status(),
        )
        logger.info("[multimodal] embed-structured report: %s", ingest_report)

        docs = chunks_to_documents(
            chunks=chunks,
            user_id=user_id,
            images_by_id=images_by_id,
            source_url_base=body.source_url_base or "",
        )

        result = await store_data_in_vector_db(
            data=docs,
            file_id=file_id,
            user_id=user_id,
            clean_content=False,
            executor=request.app.state.thread_pool,
            pre_chunked=True,
        )

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No result"
            logger.error("[multimodal] embed-structured store failed: %s", error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store multimodal chunks: {error_msg}",
            )

        stale_deleted = await cleanup_stale_vectors(
            file_id,
            ingest_run_id,
            request.app.state.thread_pool,
        )

        logger.info(
            "[multimodal] embed-structured done: file_id=%s chunks=%d stale_deleted=%d",
            file_id,
            len(chunks),
            stale_deleted,
        )
        return {
            "status": True,
            "file_id": file_id,
            "chunks_created": len(chunks),
            "images_indexed": len(images_by_id),
            "stale_vectors_deleted": stale_deleted,
            "ingest_report": ingest_report,
            "message": "Multimodal document embedded successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[multimodal] embed-structured error | file_id=%s | %s | Traceback: %s",
            file_id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during multimodal embedding: {str(e)}",
        )


@router.post("/query-multimodal")
async def query_multimodal(body: QueryRequestBody, request: Request):
    """
    Query the vector store and return context_groups with image metadata, sources, and
    context-expanded chunks (previous/next chunk linking + connected-node merging).

    Falls back to the standard flat result format when chunks lack multimodal metadata.
    """
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    try:
        search_query = normalize_search_query(body.query)
        embedding = get_cached_query_embedding(search_query)

        # Retrieve top_k hits (fetch more to allow expansion dedup)
        k_fetch = body.k * 3

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k_fetch,
                filter={"file_id": {"$eq": body.file_id}},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=k_fetch, filter={"file_id": {"$eq": body.file_id}}
            )

        retrieved_count = len(documents)
        documents = rerank_results(search_query, _apply_distance_threshold(documents))
        ranked_count = len(documents)
        multimodal_documents = [
            (doc, score)
            for doc, score in documents
            if doc.metadata.get("chunk_id")
        ]
        if multimodal_documents:
            documents = multimodal_documents
        documents = filter_ranked_results(documents, min_results=1)
        filtered_count = len(documents)

        if not documents:
            return {
                "type": "multimodal_file_search_results",
                "version": 1,
                "context_groups": [],
            }

        # Authorization check (same as /query)
        first_doc, _ = documents[0]
        doc_user_id = first_doc.metadata.get("user_id")
        if doc_user_id is not None and doc_user_id != user_authorized:
            logger.warning(
                "[multimodal] Unauthorized: user=%s doc_user=%s", user_authorized, doc_user_id
            )
            return {
                "type": "multimodal_file_search_results",
                "version": 1,
                "context_groups": [],
            }

        # Check whether chunks carry multimodal metadata
        has_multimodal = any(
            d.metadata.get("chunk_id") for d, _ in documents
        )

        logger.info(
            "[multimodal] query-multimodal: file_id=%s docs_retrieved=%d has_multimodal=%s",
            body.file_id,
            len(documents),
            has_multimodal,
        )
        if not has_multimodal and documents:
            # Show metadata keys of first doc to diagnose missing fields
            sample_meta = documents[0][0].metadata
            logger.info(
                "[multimodal] query-multimodal: first doc metadata keys=%s, chunk_id=%r",
                list(sample_meta.keys()),
                sample_meta.get("chunk_id"),
            )

        if not has_multimodal:
            # Fall back: wrap plain results in a minimal context_groups structure
            logger.debug(
                "[multimodal] query-multimodal: no chunk_id metadata, using legacy fallback"
            )
            context_groups = _legacy_to_context_groups(documents, body.file_id)
            return {
                "type": "multimodal_file_search_results",
                "version": 1,
                "context_groups": context_groups,
            }

        documents = [(d, score) for d, score in documents if d.metadata.get("chunk_id")]
        filtered_count = len(documents)
        if not documents:
            return {
                "type": "multimodal_file_search_results",
                "version": 1,
                "context_groups": [],
                "retrieval_report": {
                    "query_intent": detect_query_intent(search_query),
                    "candidates_retrieved": retrieved_count,
                    "candidates_ranked": ranked_count,
                    "candidates_after_filter": 0,
                    "groups_returned": 0,
                    "groups_with_images": 0,
                },
            }

        # Limit to top_k hits for expansion
        top_hits = documents[: body.k]
        depth_before, depth_after = _multimodal_expansion_depth(search_query, top_hits)
        all_chunks_by_id = await load_neighbor_chunks(
            body.file_id,
            [doc for doc, _ in top_hits],
            depth_before,
            depth_after,
            request.app.state.thread_pool,
        )
        groups = expand_results(
            results=top_hits,
            all_chunks_by_id=all_chunks_by_id,
            depth_before=depth_before,
            depth_after=depth_after,
        )
        merged = merge_groups(groups)
        context_groups = groups_to_context_groups(merged)
        retrieval_report = {
            "query_intent": detect_query_intent(search_query),
            "candidates_retrieved": retrieved_count,
            "candidates_ranked": ranked_count,
            "candidates_after_filter": filtered_count,
            "groups_returned": len(context_groups),
            "groups_with_images": sum(1 for group in context_groups if group.get("images")),
            "expansion_depth_before": depth_before,
            "expansion_depth_after": depth_after,
        }

        logger.debug(
            "[multimodal] query-multimodal: %d hits → %d merged groups, file_id=%s",
            len(top_hits),
            len(context_groups),
            body.file_id,
        )

        return {
            "type": "multimodal_file_search_results",
            "version": 1,
            "context_groups": context_groups,
            "retrieval_report": retrieval_report,
        }

    except HTTPException as http_exc:
        logger.error(
            "[multimodal] HTTP Exception in query-multimodal | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "[multimodal] Error in query-multimodal | file_id=%s | query=%s | %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


def _legacy_to_context_groups(
    documents: list, file_id: str
) -> list:
    """Convert legacy (doc, score) pairs into a minimal context_groups structure."""
    groups = []
    for idx, (doc, score) in enumerate(documents):
        m = doc.metadata
        source_file = m.get("source", "").split("/")[-1] or file_id
        page = m.get("page", 0)
        groups.append(
            {
                "group_id": f"group_{idx + 1}",
                "file_id": m.get("file_id", file_id),
                "source_file": source_file,
                "pages": [page] if page else [],
                "score": score,
                "chunks": [
                    {
                        "chunk_id": "",
                        "text": doc.page_content,
                        "score": score,
                        "metadata": {
                            "file_id": m.get("file_id", file_id),
                            "source_file": source_file,
                            "page": page,
                            "source_url": "",
                            "image_ids": [],
                            "images": [],
                            "previous_chunk_id": None,
                            "next_chunk_id": None,
                        },
                    }
                ],
                "images": [],
                "sources": [{"source_file": source_file, "page": page, "source_url": ""}],
            }
        )
    return groups


# ---------------------------------------------------------------------------
# PDF multimodal ingestion (figure extraction + chunking in one call)
# ---------------------------------------------------------------------------

@router.post("/embed-pdf")
async def embed_pdf_with_figures(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
    describe_images: str = Form("true"),
):
    """
    Ingest a PDF with automatic figure extraction.

    1. Saves the uploaded PDF to a temp path.
    2. Extracts raster images with pymupdf; optionally generates image_summary via OpenAI vision.
    3. Builds multimodal chunks (text + <image_id> placeholders) with figure-safe splitting.
    4. Stores chunks in the vector store.
    5. Returns image_count and chunk_count so the caller can confirm ingestion.

    Requires:
        pymupdf  — pip install pymupdf
        openai   — already available via langchain-openai; set OPENAI_API_KEY for summaries.
    """
    from app.utils.figure_extractor import extract_figures_from_pdf
    from app.utils.multimodal import chunk_page, chunks_to_documents

    user_id = get_user_id(request, entity_id)
    validated_temp_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    should_describe = describe_images.lower() not in ("false", "0", "no")

    try:
        os.makedirs(os.path.dirname(validated_temp_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_path)

        with open(validated_temp_path, "rb") as fh:
            pdf_bytes = fh.read()

        # Extract figures and per-page text (async — calls OpenAI vision per image)
        images_by_id, pages = await extract_figures_from_pdf(
            pdf_bytes=pdf_bytes,
            file_id=file_id,
            source_filename=file.filename,
            describe=should_describe,
        )

        logger.info(
            "[embed-pdf] file_id=%s pages=%d images=%d",
            file_id, len(pages), len(images_by_id),
        )

        # Build multimodal chunks from extracted pages
        loop = asyncio.get_running_loop()
        all_chunks = []
        for page_num, page_text in pages:
            page_chunks = await loop.run_in_executor(
                request.app.state.thread_pool,
                lambda pn=page_num, pt=page_text: chunk_page(
                    page_markdown=pt,
                    page_num=pn,
                    file_id=file_id,
                    source_file=file.filename,
                    images_by_id=images_by_id,
                ),
            )
            all_chunks.extend(page_chunks)

        # Link previous/next and assign sequence_number
        for i, chunk in enumerate(all_chunks):
            chunk["sequence_number"] = i + 1
            chunk["previous_chunk_id"] = all_chunks[i - 1]["chunk_id"] if i > 0 else None
            chunk["next_chunk_id"] = (
                all_chunks[i + 1]["chunk_id"] if i < len(all_chunks) - 1 else None
            )

        ingest_run_id = str(uuid.uuid4())
        for chunk in all_chunks:
            chunk["ingest_run_id"] = ingest_run_id

        domain_hints = await describe_domain_hints(
            source_file=file.filename,
            chunk_texts=[chunk.get("text", "") for chunk in all_chunks],
        )
        apply_domain_hints(all_chunks, domain_hints)
        if domain_hints:
            logger.info(
                "[embed-pdf] domain hints generated: file_id=%s domain=%s terms=%d",
                file_id,
                domain_hints.get("domain", ""),
                len(domain_hints.get("key_terms", [])),
            )

        ingest_report = build_ingest_report(
            all_chunks,
            images_by_id,
            domain_hints,
            get_domain_hints_status(),
        )
        logger.info("[embed-pdf] ingest report: %s", ingest_report)

        docs = chunks_to_documents(
            chunks=all_chunks,
            user_id=user_id,
            images_by_id=images_by_id,
        )

        result = await store_data_in_vector_db(
            data=docs,
            file_id=file_id,
            user_id=user_id,
            clean_content=False,
            executor=request.app.state.thread_pool,
            pre_chunked=True,
        )

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No result"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store PDF chunks: {error_msg}",
            )

        stale_deleted = await cleanup_stale_vectors(
            file_id,
            ingest_run_id,
            request.app.state.thread_pool,
        )

        return {
            "status": True,
            "file_id": file_id,
            "filename": file.filename,
            "chunks_created": len(all_chunks),
            "images_extracted": len(images_by_id),
            "stale_vectors_deleted": stale_deleted,
            "ingest_report": ingest_report,
            "message": "PDF embedded with figure extraction successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[embed-pdf] error | file_id=%s | %s | Traceback: %s",
            file_id, str(e), traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF figure embedding failed: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_temp_path)


# ---------------------------------------------------------------------------
# Image serving — returns extracted PNG files by file_id + image_id
# ---------------------------------------------------------------------------

@router.get("/content/image/{file_id}/{image_id}")
async def serve_extracted_image(file_id: str, image_id: str):
    """
    Serve a PNG image previously extracted from a PDF by embed-pdf.

    The path is validated to stay within RAG_UPLOAD_DIR/images/ to prevent
    path traversal.  Returns 404 if the image does not exist.
    """
    images_root = Path(RAG_UPLOAD_DIR) / "images"
    # Validate both segments — neither may contain path separators
    if "/" in file_id or "\\" in file_id or "/" in image_id or "\\" in image_id:
        raise HTTPException(status_code=400, detail="Invalid path segment")

    image_path = images_root / file_id / f"{image_id}.png"

    # Resolve and confirm the path stays inside images_root
    try:
        image_path.resolve().relative_to(images_root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"{image_id}.png",
    )
