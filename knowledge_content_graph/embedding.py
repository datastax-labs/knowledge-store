import abc
from typing import Iterable, List


class Embedding(abc.ABC):
    @property
    @abc.abstractmethod
    def dimensions(self) -> int: ...

    @abc.abstractmethod
    def embed_question(self, question: str) -> List[float]: ...

    @abc.abstractmethod
    def embed_passages(self, passages: Iterable[str]) -> List[List[float]]: ...

    @abc.abstractmethod
    def embed_passages_for_support(self, passages: Iterable[str]) -> List[List[float]]: ...

INSTRUCTION_QUESTION = "Represent the question for retrieving supporting documents"
INSTRUCTION_PASSAGE = "Represent the passage for retrieval"
INSTRUCTION_PASSAGE_FOR_SUPPORT = "Represent the passage for retrieving supporting documents"

class InstructorEmbedding(Embedding):
    def __init__(self):
        from InstructorEmbedding import INSTRUCTOR
        self.model = INSTRUCTOR('hkunlp/instructor-xl')

    @property
    def dimensions(self) -> int:
        return 768

    def embed_question(self, question: str) -> List[float]:
        return self.model.encode([[INSTRUCTION_QUESTION, question]])[0]

    def embed_passages(self, passages: Iterable[str]) -> List[List[float]]:
        return self.model.encode([[INSTRUCTION_PASSAGE, passage] for passage in passages])

    def embed_passages_for_support(self, passages: Iterable[str]) -> List[List[float]]:
        return self.model.encode([[INSTRUCTION_PASSAGE_FOR_SUPPORT, passage] for passage in passages])
