from knowledge_content_graph.content import Content
from tests.conftest import DataFixture


def test_write_retrieve(fresh_fixture: DataFixture):
    parent = Content(
        source_id="doc",
        content_id="doc-1",
        parent_id=None,
        kind="document",
        urls=["https://some_canonical_url", "https://some_other_url"],
    )
    passage_1 = Content(
        source_id="doc",
        content_id="doc-2",
        parent_id="doc-1",
        kind="passage",
        text_content="Hello World",
        keywords={"Hello", "World"},
    )
    passage_2 = Content(
        source_id="doc",
        content_id="doc-3",
        parent_id="doc-1",
        kind="passage",
        text_content="Hello Earth",
        keywords={"Hello", "Earth"},
    )
    fresh_fixture.graph.add_content([parent, passage_1, passage_2])

    results = fresh_fixture.graph.get_content_by_text("Earth")
    # Passage 2 is more similar, but World and Earth are similar enough that passage_1
    # also shows up.
    assert list(results) == [passage_2, passage_1]

    results = fresh_fixture.graph.get_content_by_url("https://some_canonical_url")
    assert list(results) == [parent]

    results = fresh_fixture.graph.get_content_by_parent("doc-1")
    assert list(results) == [passage_1, passage_2]

    results = fresh_fixture.graph.get_content_by_keyword({"Hello"})
    assert list(results) == [passage_1, passage_2]

    results = fresh_fixture.graph.get_content_by_keyword({"World"})
    assert list(results) == [passage_1]

    results = fresh_fixture.graph.get_content_by_keyword({"Earth", "World"})
    assert list(results) == [passage_1, passage_2]
