from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, queries: list, k: int):
        pass