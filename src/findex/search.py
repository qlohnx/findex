from findex.index import InvertedIndex, Posting


def intersect_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор AND для двох відсортованих списків постінгів (Two-Pointer Merge)."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            # Для AND частоту TF можна сумувати або брати з першого (для простого булевого пошуку зберігаємо doc_id)
            result.append(Posting(doc_id=p1[i].doc_id, tf=p1[i].tf + p2[j].tf))
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1

    return result


def union_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор OR для двох відсортованих списків постінгів."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result.append(Posting(doc_id=p1[i].doc_id, tf=p1[i].tf + p2[j].tf))
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1

    # Додаємо залишки
    while i < len(p1):
        result.append(p1[i])
        i += 1
    while j < len(p2):
        result.append(p2[j])
        j += 1

    return result


def difference_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор AND NOT (знаходити документи з p1, яких немає в p2)."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result.append(p1[i])
            i += 1
        else:
            j += 1

    while i < len(p1):
        result.append(p1[i])
        i += 1

    return result


class BooleanSearch:
    """Клас для виконання булевих запитів по оберненому індексу."""

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def search_and(self, term1: str, term2: str) -> list[Posting]:
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return intersect_postings(p1, p2)

    def search_or(self, term1: str, term2: str) -> list[Posting]:
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return union_postings(p1, p2)

    def search_not(self, term1: str, term2: str) -> list[Posting]:
        """Шукає term1 AND NOT term2."""
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return difference_postings(p1, p2)
