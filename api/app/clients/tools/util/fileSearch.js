const axios = require('axios');
const { logger } = require('@librechat/data-schemas');
const { tool } = require('@librechat/agents/langchain/tools');
const { generateShortLivedToken } = require('@librechat/api');
const { Tools, EToolResources } = require('librechat-data-provider');
const { filterFilesByAgentAccess } = require('~/server/services/Files/permissions');
const { getFiles } = require('~/models');

const fileSearchJsonSchema = {
  type: 'object',
  properties: {
    query: {
      type: 'string',
      description:
        "A natural language query to search for relevant information in the files. Be specific and use keywords related to the information you're looking for. The query will be used for semantic similarity matching against the file contents.",
    },
  },
  required: ['query'],
};

/**
 *
 * @param {Object} options
 * @param {ServerRequest} options.req
 * @param {Agent['tool_resources']} options.tool_resources
 * @param {string} [options.agentId] - The agent ID for file access control
 * @returns {Promise<{
 *   files: Array<{ file_id: string; filename: string }>,
 *   toolContext: string
 * }>}
 */
const primeFiles = async (options) => {
  const { tool_resources, req, agentId } = options;
  const file_ids = tool_resources?.[EToolResources.file_search]?.file_ids ?? [];
  const agentResourceIds = new Set(file_ids);
  const resourceFiles = tool_resources?.[EToolResources.file_search]?.files ?? [];

  // Get all files first
  const allFiles = (await getFiles({ file_id: { $in: file_ids } }, null, { text: 0 })) ?? [];

  // Filter by access if user and agent are provided
  let dbFiles;
  if (req?.user?.id && agentId) {
    dbFiles = await filterFilesByAgentAccess({
      files: allFiles,
      userId: req.user.id,
      role: req.user.role,
      agentId,
    });
  } else {
    dbFiles = allFiles;
  }

  dbFiles = dbFiles.concat(resourceFiles);

  let toolContext = `- Note: Semantic search is available through the ${Tools.file_search} tool but no files are currently loaded. Request the user to upload documents to search through.`;

  const files = [];
  for (let i = 0; i < dbFiles.length; i++) {
    const file = dbFiles[i];
    if (!file) {
      continue;
    }
    if (i === 0) {
      toolContext = `- Note: Use the ${Tools.file_search} tool to find relevant information within:`;
    }
    toolContext += `\n\t- ${file.filename}${
      agentResourceIds.has(file.file_id) ? '' : ' (just attached by user)'
    }`;
    files.push({
      file_id: file.file_id,
      filename: file.filename,
    });
  }

  return { files, toolContext };
};

/**
 *
 * @param {Object} options
 * @param {string} options.userId
 * @param {Array<{ file_id: string; filename: string }>} options.files
 * @param {string} [options.entity_id]
 * @param {boolean} [options.fileCitations=false] - Whether to include citation instructions
 * @returns
 */
/**
 * Converts legacy /query results (array of [docInfo, distance] pairs) into the
 * context_groups format so both code paths can be handled uniformly.
 */
const _legacyResultsToGroups = (legacyData, fileId, filename) => {
  if (!Array.isArray(legacyData)) return [];
  return legacyData.map(([docInfo, distance], idx) => {
    const meta = docInfo.metadata || {};
    const sourceFile = (meta.source || filename || '').split('/').pop() || filename;
    const page = meta.page || null;
    return {
      group_id: `legacy_${fileId}_${idx + 1}`,
      file_id: fileId,
      source_file: sourceFile,
      pages: page ? [page] : [],
      score: 1 - distance,
      chunks: [
        {
          chunk_id: '',
          text: docInfo.page_content,
          score: 1 - distance,
          metadata: {
            file_id: fileId,
            source_file: sourceFile,
            page: page,
            source_url: '',
            image_ids: [],
            images: [],
            previous_chunk_id: null,
            next_chunk_id: null,
          },
        },
      ],
      images: [],
      sources: [{ source_file: sourceFile, page: page, source_url: '' }],
    };
  });
};

/**
 * Builds a flat sources array from context_groups for the artifact metadata.
 */
const _groupsToSources = (groups, files) => {
  const fileById = Object.fromEntries(files.map((f) => [f.file_id, f.filename]));
  return groups.flatMap((group) =>
    (group.sources || []).map((src) => ({
      type: 'file',
      fileId: group.file_id,
      content: (group.chunks || []).map((c) => c.text).join('\n'),
      fileName: fileById[group.file_id] || group.source_file,
      relevance: group.score,
      pages: src.page ? [src.page] : [],
      pageRelevance: src.page ? { [src.page]: group.score } : {},
    })),
  );
};

const createFileSearchTool = async ({ userId, files, entity_id, fileCitations = false }) => {
  return tool(
    async ({ query }) => {
      if (files.length === 0) {
        return ['No files to search. Instruct the user to add files for the search.', undefined];
      }
      const jwtToken = generateShortLivedToken(userId);
      if (!jwtToken) {
        return ['There was an error authenticating the file search request.', undefined];
      }

      /**
       * @param {import('librechat-data-provider').TFile} file
       * @returns {{ file_id: string, query: string, k: number, entity_id?: string }}
       */
      const createQueryBody = (file) => {
        const body = {
          file_id: file.file_id,
          query,
          k: 5,
        };
        if (!entity_id) {
          return body;
        }
        body.entity_id = entity_id;
        logger.debug(`[${Tools.file_search}] RAG API /query body`, body);
        return body;
      };

      // Attempt multimodal query first; fall back to standard /query if unavailable or errored
      const queryFile = async (file) => {
        try {
          const res = await axios.post(
            `${process.env.RAG_API_URL}/query-multimodal`,
            createQueryBody(file),
            {
              headers: {
                Authorization: `Bearer ${jwtToken}`,
                'Content-Type': 'application/json',
              },
            },
          );
          logger.info(
            `[${Tools.file_search}] /query-multimodal response for file_id=${file.file_id}: type=${res.data?.type}, groups=${res.data?.context_groups?.length ?? 'n/a'}`,
          );
          if (res.data && res.data.type === 'multimodal_file_search_results') {
            const hasRealChunks = (res.data.context_groups || []).some((g) =>
              (g.chunks || []).some((c) => c.chunk_id),
            );
            logger.info(
              `[${Tools.file_search}] multimodal accepted for file_id=${file.file_id}, hasRealChunks=${hasRealChunks}`,
            );
            return { type: 'multimodal', data: res.data, fileIndex: files.indexOf(file) };
          }
          logger.warn(`[${Tools.file_search}] /query-multimodal returned unexpected type=${res.data?.type} for file_id=${file.file_id}`);
          return null;
        } catch (err) {
          // 404 means endpoint not deployed yet — fall back silently
          if (err?.response?.status === 404) {
            logger.debug(`[${Tools.file_search}] /query-multimodal 404 for file_id=${file.file_id} — rag_api not yet rebuilt, falling back to /query`);
          } else {
            logger.info(
              `[${Tools.file_search}] /query-multimodal error (status=${err?.response?.status}), falling back to /query: ${err?.message}`,
            );
          }
          return null;
        }
      };

      const legacyQueryFile = (file) =>
        axios
          .post(`${process.env.RAG_API_URL}/query`, createQueryBody(file), {
            headers: {
              Authorization: `Bearer ${jwtToken}`,
              'Content-Type': 'application/json',
            },
          })
          .catch((error) => {
            logger.error('Error encountered in `file_search` while querying file:', error);
            return null;
          });

      // Run multimodal queries in parallel
      const multimodalAttempts = await Promise.all(files.map(queryFile));
      const hasMultimodal = multimodalAttempts.some((r) => r !== null);
      logger.info(`[${Tools.file_search}] multimodal attempt results: hasMultimodal=${hasMultimodal}, files=${files.length}`);

      if (hasMultimodal) {
        // Merge context_groups from all files that responded with multimodal format
        const allGroups = [];
        for (let i = 0; i < files.length; i++) {
          const attempt = multimodalAttempts[i];
          if (attempt && attempt.type === 'multimodal') {
            const groups = attempt.data.context_groups || [];
            allGroups.push(...groups);
          }
        }

        // For files that did NOT return multimodal, fall back to legacy and convert
        const legacyFallbackFiles = files.filter((_, i) => !multimodalAttempts[i]);
        if (legacyFallbackFiles.length > 0) {
          const legacyResults = await Promise.all(legacyFallbackFiles.map(legacyQueryFile));
          for (let i = 0; i < legacyFallbackFiles.length; i++) {
            const result = legacyResults[i];
            if (!result) continue;
            const file = legacyFallbackFiles[i];
            const legacyGroups = _legacyResultsToGroups(result.data, file.file_id, file.filename);
            allGroups.push(...legacyGroups);
          }
        }

        if (allGroups.length === 0) {
          return [
            'No content found in the files. The files may not have been processed correctly or you may need to refine your query.',
            undefined,
          ];
        }

        const multimodalPayload = {
          type: 'multimodal_file_search_results',
          version: 1,
          context_groups: allGroups,
        };

        logger.debug(
          `[${Tools.file_search}] multimodal result: ${allGroups.length} groups`,
        );

        const toolContent = JSON.stringify(multimodalPayload);
        const sources = _groupsToSources(allGroups, files);
        return [toolContent, { [Tools.file_search]: { sources, fileCitations } }];
      }

      // Legacy path: all files use standard /query
      const queryPromises = files.map(legacyQueryFile);
      const results = await Promise.all(queryPromises);
      const validResults = results.filter((result) => result !== null);

      if (validResults.length === 0) {
        return ['No results found or errors occurred while searching the files.', undefined];
      }

      const formattedResults = validResults
        .flatMap((result, fileIndex) =>
          result.data.map(([docInfo, distance]) => ({
            filename: (docInfo.metadata?.source ?? '').split('/').pop() || files[fileIndex]?.filename || '',
            content: docInfo.page_content,
            distance,
            file_id: files[fileIndex]?.file_id,
            page: docInfo.metadata?.page || null,
          })),
        )
        .sort((a, b) => a.distance - b.distance)
        .slice(0, 10);

      if (formattedResults.length === 0) {
        return [
          'No content found in the files. The files may not have been processed correctly or you may need to refine your query.',
          undefined,
        ];
      }

      const formattedString = formattedResults
        .map(
          (result, index) =>
            `File: ${result.filename}${
              fileCitations ? `\nAnchor: \\ue202turn0file${index} (${result.filename})` : ''
            }\nRelevance: ${(1.0 - result.distance).toFixed(4)}\nContent: ${result.content}\n`,
        )
        .join('\n---\n');

      const sources = formattedResults.map((result) => ({
        type: 'file',
        fileId: result.file_id,
        content: result.content,
        fileName: result.filename,
        relevance: 1.0 - result.distance,
        pages: result.page ? [result.page] : [],
        pageRelevance: result.page ? { [result.page]: 1.0 - result.distance } : {},
      }));

      return [formattedString, { [Tools.file_search]: { sources, fileCitations } }];
    },
    {
      name: Tools.file_search,
      responseFormat: 'content_and_artifact',
      description: `Performs semantic search across attached "${Tools.file_search}" documents using natural language queries. This tool analyzes the content of uploaded files to find relevant information, quotes, and passages that best match your query. Use this to extract specific information or find relevant sections within the available documents.${
        fileCitations
          ? `

**CITE FILE SEARCH RESULTS:**
Use the EXACT anchor markers shown below (copy them verbatim) immediately after statements derived from file content. Reference the filename in your text:
- File citation: "The document.pdf states that... \\ue202turn0file0"  
- Page reference: "According to report.docx... \\ue202turn0file1"
- Multi-file: "Multiple sources confirm... \\ue200\\ue202turn0file0\\ue202turn0file1\\ue201"

**CRITICAL:** Output these escape sequences EXACTLY as shown (e.g., \\ue202turn0file0). Do NOT substitute with other characters like † or similar symbols.
**ALWAYS mention the filename in your text before the citation marker. NEVER use markdown links or footnotes.**`
          : ''
      }`,
      schema: fileSearchJsonSchema,
    },
  );
};

module.exports = { createFileSearchTool, primeFiles, fileSearchJsonSchema };
