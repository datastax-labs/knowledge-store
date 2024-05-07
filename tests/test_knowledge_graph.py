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
    passage_1 = Content(
        document_id = "doc",
        content_id = "doc-2",
        parent_id = "doc-1",
        kind = "passage",
        text_content = "Hello World",
        keywords = {"Hello", "World"}
    )
    passage_2 = Content(
        document_id = "doc",
        content_id = "doc-3",
        parent_id = "doc-1",
        kind = "passage",
        text_content = "Hello Earth",
        keywords = {"Hello", "Earth"},
    )
    fresh_fixture.graph.add_content([parent, passage_1, passage_2])

    results = fresh_fixture.graph.get_content_by_text("World")
    assert(results == [passage_1])

    results = fresh_fixture.graph.get_content_by_url("https://some_canonical_url")
    assert(results == [parent])

    results = fresh_fixture.graph.get_content_by_parent("doc-1")
    assert(results == [passage_1, passage_2])

    results = fresh_fixture.graph.get_content_by_keywords({"Hello"})
    assert(results == [passage_1, passage_2])

    results = fresh_fixture.graph.get_content_by_keywords({"World"})
    assert(results == [passage_1])

    results = fresh_fixture.graph.get_content_by_keywords({"Earth", "World"})
    assert(results == [passage_1, passage_2])