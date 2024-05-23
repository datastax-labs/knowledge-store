# Knowledge Store

Hybrid Knowledge Store combining vector similarity and edges between chunks.

## Usage

1. Create a Hybrid `KnowledgeStore` and configure how edges should be inferred.
2. Pre-process your documents to populate `metadata` information.
3. Replace your LangChain `VectorStore` with the Hybrid `KnowledgeStore`.
4. Retrieve documets from the `KnowledgeStore`.

## Metadata

The Knowledge Store makes use of the following metadata fields on each `Document`:

- `content_id`: If assigned, this specifies the unique ID of the `Document`.
  If not assigned, one will be generated.
  This should be set if you may re-ingest the same document so that it is overwritten rather than being duplicated.
- `parent_content_id`: If this `Document` is a chunk of a larger document, you may reference the parent content here.
- `keywords`: A list of strings representing keywords present in this `Document`.
- `hrefs`: A list of strings containing the URLs which this `Document` links to.
- `urls`: A list of strings containing the URLs associated with this `Document`.
  If one webpage is divided into multiple chunks, each chunk's `Document` would have the same URL.
  One webpage may have multiple URLs if it is available in multiple ways.

### Keywords

To link documents with common keywords, assign the `keywords` metadata of each `Document`.

There are various ways to assign keywords to each `Document`, such as TF-IDF across the documents.
One easy option is to use the [KeyBERT](https://maartengr.github.io/KeyBERT/index.html).

Once installed with `pip install keybert`, you can add keywords to a list `documents` as follows:

```python
from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords([doc.page_content for doc in pages],
                                     stop_words='english')

for (doc, kws) in zip(documents, keywords):
    doc.metadata["keywords"] = [kw for (kw, _distance) in kws]
```

Rather than taking all the top keywords, you could also limit to those with less than a certain `_distance` to the document.

### Links

To capture hyperlinks, populate the `hrefs` and `urls` metadata fields of each `Document`.

```python
import re
link_re = re.compile("href=\"([^\"]+)")
for doc in documents:
    doc.metadata["content_id"] = doc.metadata["source"]
    doc.metadata["hrefs"] = list(link_re.findall(doc.page_content))
    doc.metadata["urls"] = [doc.metadata["source"]]
```

## Development

```shell
poetry install --with=dev

# Run Tests
poetry run pytest
```