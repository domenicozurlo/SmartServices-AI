const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const { logger } = require('@librechat/data-schemas');
const { FileSources } = require('librechat-data-provider');
const { logAxiosError, generateShortLivedToken } = require('@librechat/api');

/**
 * Deletes a file from the vector database. This function takes a file object, constructs the full path, and
 * verifies the path's validity before deleting the file. If the path is invalid, an error is thrown.
 *
 * @param {ServerRequest} req - The request object from Express.
 * @param {MongoFile} file - The file object to be deleted. It should have a `filepath` property that is
 *                           a string representing the path of the file relative to the publicPath.
 *
 * @returns {Promise<void>}
 *          A promise that resolves when the file has been successfully deleted, or throws an error if the
 *          file path is invalid or if there is an error in deletion.
 */
const deleteVectors = async (req, file) => {
  if (!file.embedded || !process.env.RAG_API_URL) {
    return;
  }
  try {
    const jwtToken = generateShortLivedToken(req.user.id);

    return await axios.delete(`${process.env.RAG_API_URL}/documents`, {
      headers: {
        Authorization: `Bearer ${jwtToken}`,
        'Content-Type': 'application/json',
        accept: 'application/json',
      },
      data: [file.file_id],
    });
  } catch (error) {
    logAxiosError({
      error,
      message: 'Error deleting vectors',
    });
    if (
      error.response &&
      error.response.status !== 404 &&
      (error.response.status < 200 || error.response.status >= 300)
    ) {
      logger.warn('Error deleting vectors, file will not be deleted');
      throw new Error(error.message || 'An error occurred during file deletion.');
    }
  }
};

/**
 * Uploads a file to the configured Vector database
 *
 * @param {Object} params - The params object.
 * @param {Object} params.req - The request object from Express. It should have a `user` property with an `id` representing the user
 * @param {Express.Multer.File} params.file - The file object, which is part of the request. The file object should
 *                                     have a `path` property that points to the location of the uploaded file.
 * @param {string} params.file_id - The file ID.
 * @param {string} [params.entity_id] - The entity ID for shared resources.
 * @param {Object} [params.storageMetadata] - Storage metadata for dual storage pattern.
 *
 * @returns {Promise<{ filepath: string, bytes: number }>}
 *          A promise that resolves to an object containing:
 *            - filepath: The path where the file is saved.
 *            - bytes: The size of the file in bytes.
 */
async function uploadVectors({ req, file, file_id, entity_id, storageMetadata }) {
  if (!process.env.RAG_API_URL) {
    throw new Error('RAG_API_URL not defined');
  }

  try {
    const jwtToken = generateShortLivedToken(req.user.id);
    const formData = new FormData();
    formData.append('file_id', file_id);
    formData.append('file', fs.createReadStream(file.path));
    if (entity_id != null && entity_id) {
      formData.append('entity_id', entity_id);
    }

    // Include storage metadata for RAG API to store with embeddings
    if (storageMetadata) {
      formData.append('storage_metadata', JSON.stringify(storageMetadata));
    }

    const formHeaders = formData.getHeaders();

    const response = await axios.post(`${process.env.RAG_API_URL}/embed`, formData, {
      headers: {
        Authorization: `Bearer ${jwtToken}`,
        accept: 'application/json',
        ...formHeaders,
      },
    });

    const responseData = response.data;
    logger.debug('Response from embedding file', responseData);

    if (responseData.known_type === false) {
      throw new Error(`File embedding failed. The filetype ${file.mimetype} is not supported`);
    }

    if (!responseData.status) {
      throw new Error('File embedding failed.');
    }

    return {
      bytes: file.size,
      filename: file.originalname,
      filepath: FileSources.vectordb,
      embedded: Boolean(responseData.known_type),
    };
  } catch (error) {
    logAxiosError({
      error,
      message: 'Error uploading vectors',
    });
    throw new Error(error.message || 'An error occurred during file upload.');
  }
}

/**
 * Uploads structured multimodal OCR data to the RAG API for multimodal chunking and embedding.
 * Falls back silently if RAG_API_URL is not configured or the endpoint is unavailable.
 *
 * @param {Object} params
 * @param {Object} params.req - Express request object with user info
 * @param {string} params.file_id - The file ID
 * @param {Object} params.structured_ocr - Structured OCR result from processOCRResultStructured
 * @param {string} [params.entity_id] - Optional entity ID for shared resources
 * @returns {Promise<void>}
 */
async function uploadStructuredVectors({ req, file_id, structured_ocr, entity_id, source_url_base }) {
  if (!process.env.RAG_API_URL) {
    logger.warn('[multimodal] RAG_API_URL not set, skipping structured embed');
    return;
  }
  if (!structured_ocr) {
    return;
  }
  try {
    const jwtToken = generateShortLivedToken(req.user.id);
    const payload = {
      file_id,
      structured_ocr,
      ...(entity_id ? { entity_id } : {}),
      ...(source_url_base ? { source_url_base } : {}),
    };
    logger.debug(
      `[multimodal] Posting structured OCR to /embed-structured: file_id=${file_id}, pages=${structured_ocr.pages?.length ?? 0}`,
    );
    const response = await axios.post(`${process.env.RAG_API_URL}/embed-structured`, payload, {
      headers: {
        Authorization: `Bearer ${jwtToken}`,
        'Content-Type': 'application/json',
        accept: 'application/json',
      },
    });
    logger.debug('[multimodal] /embed-structured response:', response.data);
  } catch (error) {
    // Log but do not throw — multimodal indexing is best-effort and must not
    // break the primary OCR upload flow.
    logAxiosError({ error, message: '[multimodal] Error posting to /embed-structured' });
    logger.warn('[multimodal] Structured embed failed, RAG will fall back to plain text search');
  }
}

module.exports = {
  deleteVectors,
  uploadVectors,
  uploadStructuredVectors,
};
