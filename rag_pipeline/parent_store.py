"""Parent document store backed by Elasticsearch.

Stores full parent document text keyed by ``parent_id`` so that child
chunks can retrieve their full parent context during query time
(RAGFlow-style parent-child chunking).

Design:
  - Lazy client initialization: no Elasticsearch connection at import
    time; the client is created on first method call.
  - Graceful degradation: any Elasticsearch failure (unreachable,
    timeout, request error) is logged and the public methods no-op /
    return None. Public methods never raise.
  - Feature gating (``PARENT_CHILD_ENABLED``) lives at call sites, not
    inside this store.
"""

import logging
from typing import Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from config import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_CLOUD_ID,
    ELASTICSEARCH_API_KEY,
    PARENT_CHUNK_MAX_SIZE,
    PARENT_STORE_ES_INDEX,
)

logger = logging.getLogger(__name__)


class ParentStore:
    """Elasticsearch-backed store for full parent document text."""

    def __init__(
        self,
        es_url: Optional[str] = None,
        index_name: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.es_url = es_url or ELASTICSEARCH_URL
        self.index_name = index_name or PARENT_STORE_ES_INDEX
        self.cloud_id = cloud_id if cloud_id is not None else ELASTICSEARCH_CLOUD_ID
        self.api_key = api_key if api_key is not None else ELASTICSEARCH_API_KEY
        self._client: Optional[Elasticsearch] = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Elasticsearch]:
        """Return the ES client, connecting lazily on first use."""
        if self._client is None:
            try:
                # Prefer cloud credentials when available (CI/remote runs),
                # fall back to a plain URL (local docker ES).
                if self.cloud_id and self.api_key:
                    self._client = Elasticsearch(
                        cloud_id=self.cloud_id,
                        api_key=self.api_key,
                        request_timeout=5,
                    )
                else:
                    # Short timeout so ES-down degradation fails fast instead
                    # of hanging on the client's default retry/backoff.
                    self._client = Elasticsearch(self.es_url, request_timeout=5)
                self._ensure_index()
            except Exception as e:
                logger.warning(
                    "ParentStore: Elasticsearch unavailable at %s: %s",
                    self.es_url,
                    e,
                )
                self._client = None
        return self._client

    def _ensure_index(self) -> None:
        """Create the parent index if it does not exist."""
        if self._client.indices.exists(index=self.index_name):
            return
        try:
            self._client.indices.create(index=self.index_name)
            logger.info("ParentStore: created index '%s'", self.index_name)
        except Exception as e:
            # Concurrent creation races are fine (resource_already_exists,
            # HTTP 400); any other failure propagates to _get_client.
            if getattr(e, "status_code", None) != 400:
                raise
            logger.info("ParentStore: index '%s' already exists", self.index_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put_parent(self, parent_id: str, content: str) -> None:
        """Index/upsert a parent document.

        Content longer than ``PARENT_CHUNK_MAX_SIZE`` characters is
        truncated before storage. If Elasticsearch is unavailable the
        write is skipped with a warning.
        """
        client = self._get_client()
        if client is None:
            logger.warning(
                "ParentStore: ES unavailable, skipping put_parent(%s)", parent_id
            )
            return
        try:
            truncated = content[:PARENT_CHUNK_MAX_SIZE]
            client.index(
                index=self.index_name,
                id=parent_id,
                document={"parent_id": parent_id, "content": truncated},
            )
        except Exception as e:
            logger.warning(
                "ParentStore: put_parent(%s) failed: %s", parent_id, e
            )

    def get_parent(self, parent_id: str) -> Optional[str]:
        """Fetch parent content by id; None if missing or ES unavailable."""
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.get(index=self.index_name, id=parent_id)
            return resp.get("_source", {}).get("content")
        except NotFoundError:
            return None
        except Exception as e:
            logger.warning(
                "ParentStore: get_parent(%s) failed: %s", parent_id, e
            )
            return None

    def delete_parent(self, parent_id: str) -> None:
        """Delete a parent document by id; no-op if missing or ES down."""
        client = self._get_client()
        if client is None:
            logger.warning(
                "ParentStore: ES unavailable, skipping delete_parent(%s)",
                parent_id,
            )
            return
        try:
            client.delete(index=self.index_name, id=parent_id)
        except NotFoundError:
            pass
        except Exception as e:
            logger.warning(
                "ParentStore: delete_parent(%s) failed: %s", parent_id, e
            )
