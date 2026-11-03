from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Posting:
    """Запис про одне входження терміна в документ."""

    doc_id: int
    tf: int  # Term Frequency (скільки разів слово зустрілося в документі)


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Метадані документа для відображення у результатах пошуку."""

    path: str


class InvertedIndex:
    """Обернений індекс для швидкого пошуку документів за термінами."""

    def __init__(self) -> None:
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}

    def add_document(self, doc_id: int, path: str, tokens: list[str]) -> None:
        """Додає один документ до індексу."""
        self.doc_meta[doc_id] = DocMeta(path=path)
        self.doc_lengths[doc_id] = len(tokens)

        counts = Counter(tokens)

        for term, tf in counts.items():
            self.index[term].append(Posting(doc_id=doc_id, tf=tf))

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає список постінгів для терміна."""
        return self.index.get(term, [])

    def __len__(self) -> int:
        """Повертає кількість унікальних слів в індексі."""
        return len(self.index)
