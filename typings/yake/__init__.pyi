from typing import List, Tuple

class KeywordExtractor:
    def __init__(
        self,
        *,
        lan: str | None = ...,
        top: int | None = ...,
        n: int | None = ...,
        dedupLim: float | None = ...,
    ) -> None: ...
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]: ...
