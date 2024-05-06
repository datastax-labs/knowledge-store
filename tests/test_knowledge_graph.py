from knowledge_content_graph.content import Content
from tests.conftest import DataFixture

def test_write_retrieve(fresh_fixture: DataFixture):
    parent = Content(
        document_id = "doc",
        content_id = "doc-1",
        parent_id = None,
        kind = "document",
        urls = ["https://some_canonical_url", "https://some_other_url"]
    )
    passage = Content(
        document_id = "doc",
        content_id = "doc-2",
        parent_id = "doc-1",
        kind = "passage",
        text_content = "Hello World",
    )
    fresh_fixture.graph.add_content([parent, passage])

    results = fresh_fixture.graph.get_content_by_text("world")
    assert(results == [passage])

    results = fresh_fixture.graph.get_content_by_url("https://some_canonical_url")
    assert(results == [parent])

    results = fresh_fixture.graph.get_content_by_parent("doc-1")
    assert(results == [passage])