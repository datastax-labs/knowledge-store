from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from cassandra.cluster import Session, ResponseFuture
from cassandra.query import BatchStatement
from cassio.config import check_resolve_keyspace, check_resolve_session

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from knowledge_content_graph.content import Content
from knowledge_content_graph._utils import batched
from knowledge_content_graph.embedding import Embedding

def _results_to_content(results: ResponseFuture) -> Iterable[Content]:
    for c in results:
        yield Content(
            source_id=c.source_id,
            content_id=c.content_id,
            parent_id=c.parent_id,
            kind=c.kind,
            keywords=set(c.keywords) if c.keywords else set(),
            urls=set(c.urls) if c.urls else set(),
            links=set(c.links) if c.links else set(),
            text_content=c.text_content,
        )
class ContentGraph:
    def __init__(
        self,
        text_embedding: Embedding,
        content_table: str = "content",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        apply_schema: bool = True,
    ) -> None:
        """
        Create a Cassandra-backed Content Graph.

        Parameters:
        - content_table: Name of the table to use for content. Defaults to `"content"`.
        - session: The Cassandra `Session` to use. If not specified, uses the default `cassio`
          session, which requires `cassio.init` has been called.
        - keyspace: The Cassandra keyspace to use. If not specified, uses the default `cassio`
          keyspace, which requires `cassio.init` has been called.
        - apply_schema: If true, the node table and edge table are created.
        """

        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        self._session = session
        self._keyspace = keyspace

        self._content_table = content_table
        self._text_embedding = text_embedding

        if apply_schema:
            self._apply_schema()

        self._insert_content = self._session.prepare(
            f"""
            INSERT INTO {keyspace}.{content_table} (
                source_id, content_id, parent_id, kind, keywords, urls, links, text_content, text_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )

        SELECT = f"""
        SELECT source_id, content_id, parent_id, kind, keywords, urls, links, text_content
        FROM {keyspace}.{content_table}
        """

        SELECT_IDS = f"""
        SELECT source_id, content_id, parent_id
        FROM {keyspace}.{content_table}
        """

        self._query_text_embedding = self._session.prepare(
            f"{SELECT} ORDER BY text_embedding ANN OF ? LIMIT ?"
        )

        self._query_ids_by_keyword = self._session.prepare(
            f"{SELECT_IDS} WHERE keywords CONTAINS ?"
        )

        self._query_by_url = self._session.prepare(
            f"{SELECT} WHERE urls CONTAINS ?"
        )

        self._query_by_link = self._session.prepare(
            f"{SELECT} WHERE links CONTAINS ?"
        )

        self._query_by_parent_id = self._session.prepare(
            f"{SELECT} WHERE parent_id = ?"
        )

        self._query_by_content_ids = self._session.prepare(
            f"{SELECT} WHERE source_id = ? AND content_id IN ?"
        )

    def _apply_schema(self):
        text_embedding_dim = self._text_embedding.dimensions
        print(f"Text Dim: {text_embedding_dim}")

        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._content_table} (
                -- Content ID of the document this is part of.
                source_id TEXT,
                -- ID of this content. If this is a document, the content_id == source_id.
                content_id TEXT,
                -- The ID of parent content, if any.
                parent_id TEXT,
                -- The content kind. This should be one of the supported / known kinds.
                kind TEXT,
                -- Keywords associated with the chunk.
                keywords set<text>,
                -- One or more URLs associated with the content. This allows providing
                -- multiple URLs for documents that may have multiple locations.
                urls SET<text>,
                -- Set of URLs the content links to.
                links SET<text>,
                -- If the content kind is a text passage or otherwise contains text, this
                -- is the corresponding text content.
                text_content TEXT,
                text_embedding VECTOR<FLOAT, {text_embedding_dim}>,

                -- Partition by source ID. This allows retrieval of all content from a single
                -- document within a single partition.
                PRIMARY KEY (source_id, content_id)
            );
            """
        )

        # Index on text_embedding (for similarity)
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_text_embedding_index
            ON {self._keyspace}.{self._content_table}(text_embedding)
            USING 'sai';
            """
        )

        # Index on content_id.
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_content_id_index
            ON {self._keyspace}.{self._content_table} (content_id)
            USING 'sai';
            """
        )

        # Index on keywords
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_keywords_index
            ON {self._keyspace}.{self._content_table} (keywords)
            USING 'sai';
            """
        )

        # Index on url
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_url_index
            ON {self._keyspace}.{self._content_table} (urls)
            USING 'sai';
            """
        )

        # Index on links
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_links_index
            ON {self._keyspace}.{self._content_table} (links)
            USING 'sai';
            """
        )

        # Index on parent_id
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._content_table}_parent_index
            ON {self._keyspace}.{self._content_table} (parent_id)
            USING 'sai';
            """
        )

    def _document_to_content(self, doc: Document) -> Content:
        urls = doc.metadata.get("urls", None)
        if urls is None:
            url = doc.metadata.get("url", None)
            if url is None:
                urls = set()
            else:
                urls = {url}

        return Content(
            source_id = doc.metadata["source_id"],
            document_id = doc.metadata["document_id"],
            parent_id = doc.metadata.get("parent_id", None),
            kind = doc.metadata.get("kind", "passage"),
            keywords = doc.metadata.get("keywords", set()),
            links = doc.metadata.get("links", set()),
            urls = urls,
            text_content = doc.page_content
        )

    def add_documents(self, documents: Iterable[Document]):
        """
        Loads the content of each document as a separate entry.

        The following metadata fields are used specially:
        - `source_id`: The ID of the source of this document.
        - `document_id`: The ID of this document.
        - `kind` (optional): The kind of content this document represents.
          If not set, defaults to `"passage"`.
        - `parent_id` (optional): The ID this document is structurally part of.
          If not specified, this is a top-level section.
        - `url` or `urls` (optional): if present, contain the URLs of the
          document itself. This may be a list or set, allowing for documents
          which are available at multiple paths to be properly linked.
        - `links` (optional): if present, contain the URLs linked to by this document.
        - `keywords` (optional): if present, a set of keywords associated with
          the content. Each content will be linked to other chunks with the same
          keywords, so these should be specific to the contet (rather than the
          document). These may be extracted by TF-IDF.

        Edges will be automatically added from a document A to the document B:
        - If document A links to document B, or
        - If there are overlapping keywords between document A and document B, or
        - If MDR embedding suggests document B is similar to (supports) document A.
        """
        contents = [self._document_to_content(doc) for doc in documents]

        for document in documents:
            # TODO: Change the embedding to use MDR (directed) embedding.
            pass
        pass

    def add_content(self, contents: Iterable[Content]):
        for batch in batched(contents, n=50):
            batch_statement = BatchStatement()

            text_contents = [c.text_content for c in batch if c.text_content]
            text_contents_embedding = iter(self._text_embedding.embed_passages(text_contents))
            for content in batch:
                batch_statement.add(
                    self._insert_content,
                    (
                        content.source_id,
                        content.content_id,
                        content.parent_id,
                        content.kind,
                        content.keywords,
                        content.urls,
                        content.links,
                        content.text_content,
                        next(text_contents_embedding) if content.text_content else None,
                    ),
                )

            # TODO: Support concurrent execution of these statements.
            self._session.execute(batch_statement)
        pass

    def get_content_by_text(self, question: str, k: int = 10) -> Iterable[Content]:
        question_embedding = self._text_embedding.embed_question(question)
        results = self._session.execute(self._query_text_embedding, (question_embedding, k))
        return _results_to_content(results)

    def get_content_by_url(self, url: str) -> Iterable[Content]:
        results = self._session.execute(self._query_by_url, (url, ))
        return _results_to_content(results)

    def get_content_by_parent(self, parent_id: str) -> Iterable[Content]:
        results = self._session.execute(self._query_by_parent_id, (parent_id,))
        return _results_to_content(results)

    def get_content_by_keyword(self, keywords: Union[str, Iterable[str]]) -> Iterable[Content]:
        keyword_list = []
        if isinstance(keywords, str):
            keyword_list = [keywords]
        else:
            keyword_list = list(keywords)

        # TODO: Make async
        source_content_ids: Dict[str, Set[str]] = {}
        for keyword in keyword_list:
            for row in self._session.execute(self._query_ids_by_keyword, (keyword,)):
                source_content_ids.setdefault(row.source_id, set()).add(row.content_id)

        return self._get_content_by_ids(source_content_ids)

    def _get_content_by_ids(self, source_content_ids: Dict[str, Iterable[str]]) -> Iterable[Content]:
        # TODO: Make async
        for source_id, content_ids in source_content_ids.items():
            rows = self._session.execute(self._query_by_content_ids, (source_id, content_ids))
            for content in _results_to_content(rows):
                yield content