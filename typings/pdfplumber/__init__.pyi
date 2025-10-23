from typing import BinaryIO, ContextManager, Iterable, Protocol

class Page(Protocol):
    def extract_text(self) -> str | None: ...

class PDF:
    pages: Iterable[Page]

def open(stream: BinaryIO) -> ContextManager[PDF]: ...
