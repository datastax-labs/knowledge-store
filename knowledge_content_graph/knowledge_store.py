from itertools import repeat
import threading
from typing import Any, Dict, Iterable, List, Optional, Self, Set, Union
import uuid

from cassandra.cluster import Session, ResponseFuture
from cassandra.query import BatchStatement
from cassio.config import check_resolve_keyspace, check_resolve_session

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

def _results_to_documents(results: ResponseFuture) -> Iterable[Document]:
    for row in results:
        yield Document(
            page_content = row.text_content,
            metadata = {
                "content_id": row.content_id,
                "kind": row.kind,
            }
        )

class KnowledgeStore(VectorStore):
    """A hybrid vector-and-graph knowledge store.

    Document chunks support vector-similarity search as well as edges linking
    chunks based on structural and semantic properties.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        table: str = "knowledge",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        apply_schema: bool = True,
        concurrency: int = 20,
        infer_links: Union[bool, Set[str]] = True,
        infer_keywords: Union[bool, Set[str]] = True,
    ):
        """
        Create the hybrid knowledge store.

        Parameters configure the ways that edges should be added between
        documents. Many take `Union[bool, Set[str]]`, with `False` disabling
        inference, `True` enabling it globally between all documents, and a set
        of metadata fields defining a scope in which to enable it. Specifically,
        passing a set of metadata fields such as `source` only links documents
        with the same `source` metadata value.

        Args:
            embedding: The embeddings to use for the document content.
            concurrency: Maximum number of queries to have concurrently executing.
            apply_schema: If true, the schema will be created if necessary. If false,
                the schema must have already been applied.
            infer_links: Whether to enable (and optionally scope for) inference
                based on the `hrefs` and `urls` in the metadata. These metadata
                fields should be populated with a collection of URLs referenced
                by the document (hrefs) and a collection of URLs representing
                the document (urls), respectively.
            infer_keywords: Whether to enable (and optionally scope for)
                inference based on the `keywords` in the metadata. This metadata
                should be populated with a collection of keywords present in the
                document.
        """
        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        self._concurrency = concurrency
        self._embedding = embedding
        self._table = table
        self._session = session
        self._keyspace = keyspace

        if apply_schema:
            self._apply_schema()

        self._infer_links = infer_links
        self._infer_keywords = infer_keywords

        # TODO: Metadata
        # TODO: Parent ID / source ID / etc.
        self._insert_passage = session.prepare(
            f"""
            INSERT INTO {keyspace}.{table} (
                content_id, kind, text_content, text_embedding
            ) VALUES (?, 'passage', ?, ?)
            """
        )

        self._query_by_embedding = session.prepare(
            f"""
            SELECT content_id, kind, text_content
            FROM {keyspace}.{table}
            ORDER BY text_embedding ANN OF ?
            LIMIT ?
            """
        )

    def _apply_schema(self):
        """Apply the schema to the database."""
        embedding_dim = len(self._embedding.embed_query("Test Query"))
        self._session.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table} (
                content_id TEXT,
                kind TEXT,
                text_content TEXT,
                text_embedding VECTOR<FLOAT, {embedding_dim}>,

                PRIMARY KEY (content_id)
            )
            """
        )

        # Index on text_embedding (for similarity search)
        self._session.execute(
            f"""CREATE CUSTOM INDEX IF NOT EXISTS {self._table}_text_embedding_index
            ON {self._keyspace}.{self._table}(text_embedding)
            USING 'sai';
            """
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        metadatas = repeat(None) if metadatas is None else iter(metadatas)
        ids = [uuid.uuid4().hex for _ in texts] if ids is None else ids
        text_embeddings = self._embedding.embed_documents(texts)

        pending = len(texts)
        pending_lock = threading.Lock()
        tuples = zip(texts, text_embeddings, metadatas, ids, strict=True)

        failure = None
        event = threading.Event()

        def send_query() -> Optional[ResponseFuture]:
            if event.is_set():
                # This happens if we've recorded an error (early termination).
                # No need to send more queries since we're reporting failure.
                return None
            else:
                try:
                    text, text_embedding, metadata, id = next(tuples)
                    return self._session.execute_async(
                        self._insert_passage, (id, text, text_embedding)
                    )
                except StopIteration:
                    return None

        def handle_result(_result):
            nonlocal pending
            with pending_lock:
                pending -= 1
                if pending == 0:
                    event.set()
                    return

            query = send_query()
            if query is not None:
                attach_callback(query)

        def handle_error(err):
            nonlocal failure
            print(f"Failed: {err}")
            failure = err
            event.set()

        def attach_callback(future: ResponseFuture):
            future.add_callbacks(handle_result, handle_error)

        initial_queries = [send_query()
                           for _ in range(0, min(len(texts), self._concurrency))]
        for query in initial_queries:
            attach_callback(query)
        event.wait()

        if failure:
            raise failure
        with pending_lock:
            assert pending == 0

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Self:
        """Return VectorStore initialized from texts and embeddings."""

        knowledge_store = KnowledgeStore(embedding, **kwargs)
        knowledge_store.add_texts(texts, metadatas)
        return knowledge_store

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        metadata_filter: Dict[str, str] = {},
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
        Returns:
            List of Document, the most similar to the query vector.
        """
        embedding_vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k=k,
            metadata_filter=metadata_filter,
        )

    def similarity_search_by_vector(
            self,
            query_vector: List[float],
            *,
            k: int = 4,
            metadata_filter: Dict[str, str] = {},
    ) -> List[Document]:
        """Retun docs most similar to query_vector.

        Args:
            query_vector: Embeding to lookup documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
        Returns:
            List of Document, the most simliar to the query vector.
        """
        results = self._session.execute(self._query_by_embedding, (query_vector, k))
        return _results_to_documents(results)
