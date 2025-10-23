from typing import BinaryIO, Iterable, Protocol

class Paragraph(Protocol):
    text: str

class DocxDocument:
    paragraphs: Iterable[Paragraph]

def Document(docx: BinaryIO | str | None = ...) -> DocxDocument: ...
