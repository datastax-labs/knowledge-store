import abc
from typing import Iterable, List


class Embedding(abc.ABC):
    @property
    @abc.abstractmethod
    def dimensions(self) -> int: ...

    @abc.abstractmethod
    def embed_question(self, question: str) -> List[float]:
        """Return the embedding of the question for retrieving supporting documents.

        This should be used on a question to find passages that support it.
        """
        ...

    @abc.abstractmethod
    def embed_passages(self, passages: Iterable[str]) -> List[List[float]]:
        """Return the embedding of a passage for retrieval.

        This should be applied to passages being stored in the vector store.
        """
        ...

    @abc.abstractmethod
    def embed_passages_for_support(self, passages: Iterable[str]) -> List[List[float]]:
        """Return the emedding of the passage for retrieving supporting documents.

        This should be applied to a passage to find other supporting passages.
        """
        ...


INSTRUCTION_QUESTION = "Represent the question for retrieving supporting documents"
INSTRUCTION_PASSAGE = "Represent the passage for retrieval"
INSTRUCTION_PASSAGE_FOR_SUPPORT = "Represent the passage for retrieving supporting documents"


class InstructorEmbedding(Embedding):
    def __init__(self):
        from InstructorEmbedding import INSTRUCTOR

        self.model = INSTRUCTOR("hkunlp/instructor-xl")

    @property
    def dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_question(self, question: str) -> List[float]:
        return self.model.encode([[INSTRUCTION_QUESTION, question]])[0]

    def embed_passages(self, passages: Iterable[str]) -> List[List[float]]:
        return self.model.encode([[INSTRUCTION_PASSAGE, passage] for passage in passages])

    def embed_passages_for_support(self, passages: Iterable[str]) -> List[List[float]]:
        return self.model.encode(
            [[INSTRUCTION_PASSAGE_FOR_SUPPORT, passage] for passage in passages]
        )
