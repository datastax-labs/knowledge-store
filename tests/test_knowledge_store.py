from tests.conftest import DataFixture
from langchain_core.documents import Document

def test_write_retrieve(fresh_fixture: DataFixture):
    greetings = Document(
        page_content="Typical Greetings",
        metadata = {
            "content_id": "greetings",
        }
    )
    doc1 = Document(
        page_content="Hello World",
        metadata = {
            "content_id": "doc1",
            "parent_content_id": "greetings",
        }
    )
    doc2 = Document(
        page_content="Hello Earth",
        metadata = {
            "content_id": "doc2",
            "parent_content_id": "greetings",
        }
    )

    store = fresh_fixture.store([greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also shows up.
    results = store.similarity_search("Earth", k=2)
    assert list(map(lambda d: d.page_content, results)) == [doc2.page_content, doc1.page_content]

    results = store.retrieve("Earth", k=2, depth=0)
    assert set(map(lambda d: d.page_content, results)) == {doc2.page_content, doc1.page_content}

    results = store.retrieve("Earth", k=2, depth=1)
    assert set(map(lambda d: d.page_content, results)) == {doc2.page_content, doc1.page_content, greetings.page_content}