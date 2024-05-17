from tests.conftest import DataFixture
from langchain_core.documents import Document

def test_write_retrieve(fresh_fixture: DataFixture):
    doc1 = Document(
        page_content="Hello World",
    )
    doc2 = Document(
        page_content="Hello Earth"
    )

    store = fresh_fixture.store([doc1, doc2])

    results = store.similarity_search("Earth")
    # Doc2 is more similar, but World and Earth are similar enough that doc1
    # also shows up.
    assert list(map(lambda d: d.page_content, results)) == [doc2.page_content, doc1.page_content]
