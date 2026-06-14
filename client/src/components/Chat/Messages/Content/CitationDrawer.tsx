import { useState, useEffect, useRef, useCallback } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { X, Download } from 'lucide-react';
import { useFileDownload } from '~/data-provider';
import { useLocalize } from '~/hooks';
import store from '~/store';

export default function CitationDrawer() {
  const [citation, setCitation] = useRecoilState(store.citationPanel);
  const user = useRecoilValue(store.user);
  const localize = useLocalize();

  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const loadingRef = useRef(false);
  const prevFileIdRef = useRef<string | null>(null);

  const { refetch: downloadFile } = useFileDownload(user?.id ?? '', citation?.fileId, {
    direct: false,
  });

  const load = useCallback(async () => {
    if (!citation?.fileId || loadingRef.current) {
      return;
    }
    loadingRef.current = true;
    setLoading(true);
    setError(false);
    try {
      const result = await downloadFile();
      if (!result.data) {
        setError(true);
        return;
      }
      const resp = await fetch(result.data);
      const blob = await resp.blob();
      const typed = new Blob([blob], { type: 'application/pdf' });
      const url = URL.createObjectURL(typed);
      setBlobUrl(url);
    } catch {
      setError(true);
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, [citation?.fileId, downloadFile]);

  useEffect(() => {
    if (!citation) {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        setBlobUrl(null);
      }
      setError(false);
      setLoading(false);
      prevFileIdRef.current = null;
      return;
    }
    if (citation.fileId !== prevFileIdRef.current) {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        setBlobUrl(null);
      }
      prevFileIdRef.current = citation.fileId;
      load();
    }
  }, [citation, blobUrl, load]);

  useEffect(() => {
    return () => {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [blobUrl]);

  const handleDownload = useCallback(async () => {
    if (!citation?.fileId) {
      return;
    }
    try {
      const result = await downloadFile();
      if (!result.data) {
        return;
      }
      const a = document.createElement('a');
      a.href = result.data;
      a.download = citation.fileName;
      a.click();
    } catch {
      // ignore
    }
  }, [citation, downloadFile]);

  const isOpen = citation != null;

  return (
    <div
      className={[
        'fixed right-0 top-0 z-50 flex h-full flex-col bg-surface-primary shadow-2xl transition-transform duration-300 ease-in-out',
        isOpen ? 'translate-x-0' : 'translate-x-full',
      ].join(' ')}
      style={{ width: '42rem', maxWidth: '95vw' }}
      aria-label={localize('com_ui_preview')}
      role="complementary"
    >
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-border-light px-4 py-3">
        <div className="min-w-0 flex-1 pr-4">
          <p className="truncate text-sm font-medium text-text-primary">{citation?.fileName}</p>
          {citation?.page && (
            <p className="mt-0.5 text-xs text-text-secondary">
              {localize('com_ui_page')} {citation.page}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {citation?.fileId && (
            <button
              type="button"
              onClick={handleDownload}
              className="rounded p-1.5 text-text-secondary transition-colors hover:bg-surface-hover hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border-heavy"
              aria-label={localize('com_ui_download')}
            >
              <Download className="size-4" aria-hidden="true" />
            </button>
          )}
          <button
            type="button"
            onClick={() => setCitation(null)}
            className="rounded p-1.5 text-text-secondary transition-colors hover:bg-surface-hover hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border-heavy"
            aria-label={localize('com_ui_close')}
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="relative flex-1 overflow-hidden">
        {loading && (
          <div className="flex h-full items-center justify-center">
            <div className="size-8 animate-spin rounded-full border-4 border-border-light border-t-text-secondary" />
          </div>
        )}
        {error && !loading && (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-text-secondary">{localize('com_ui_preview_unavailable')}</p>
          </div>
        )}
        {blobUrl && !loading && (
          <iframe
            src={citation?.page ? `${blobUrl}#page=${citation.page}` : blobUrl}
            title={citation?.fileName ?? ''}
            className="h-full w-full border-0"
          />
        )}
      </div>
    </div>
  );
}
