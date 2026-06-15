import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, queries: list, k: int):
        pass