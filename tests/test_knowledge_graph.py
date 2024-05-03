from knowledge_content_graph.content import Content
from tests.conftest import DataFixture

def test_write_retrieve(fresh_fixture: DataFixture):
    fresh_fixture.graph.add_content([
        Content(
            document_id = "doc",
            content_id = "doc-1",
            parent_id = None,
            kind = "document",
        ),
        Content(
            document_id = "doc",
            content_id = "doc-2",
            parent_id = "doc-1",
            kind = "passage",
            text_content = "Hello World",
        )
    ])

    results = fresh_fixture.graph.get_content_by_text("world")
    assert(results == [
        Content(
            document_id = "doc",
            content_id = "doc-2",
            parent_id = "doc-1",
            kind = "passage",
            text_content = "Hello World",
        )
    ])