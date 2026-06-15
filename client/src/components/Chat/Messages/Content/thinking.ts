const THINKING_START = ':::thinking';
const THINKING_BLOCK_RE = /:::thinking([\s\S]*?):::/g;

export const parseTransientThinking = (text: string, allowPartial = false) => {
  const thinkingParts: string[] = [];
  THINKING_BLOCK_RE.lastIndex = 0;

  let regularContent = text
    .replace(THINKING_BLOCK_RE, (_match, content) => {
      const trimmed = String(content).trim();
      if (trimmed.length > 0) {
        thinkingParts.push(trimmed);
      }
      return '';
    })
    .trim();

  const thinkingStart = allowPartial ? regularContent.indexOf(THINKING_START) : -1;
  if (thinkingStart !== -1) {
    const partialThinking = regularContent
      .slice(thinkingStart + THINKING_START.length)
      .replace(/^\n/, '')
      .trim();
    if (partialThinking.length > 0) {
      thinkingParts.push(partialThinking);
    }
    regularContent = regularContent.slice(0, thinkingStart).trim();
  }

  return {
    thinkingContent: thinkingParts.join('\n\n'),
    regularContent,
  };
};
