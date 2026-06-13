from abc import ABC, abstractmethod
from typing import Iterator, List

from .types import LLMResponse, Message


class BaseLLMClient(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse: ...

    @abstractmethod
    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]: ...
