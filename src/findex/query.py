"""A small recursive-descent query language for the inverted index."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize


class Query:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def __and__(self, other: Query) -> Query:
        return And(self, other)

    def __or__(self, other: Query) -> Query:
        return Or(self, other)

    def __invert__(self) -> Query:
        return Not(self)

    def terms(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class Term(Query):
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", next(iter(tokenize(self.value)), ""))

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get_postings(self.value)}

    def terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True)
class Phrase(Query):
    values: tuple[str, ...] | str

    def __post_init__(self) -> None:
        values = self.values if isinstance(self.values, tuple) else tuple(tokenize(self.values))
        object.__setattr__(self, "values", values)

    def evaluate(self, index: Index) -> set[int]:
        if not self.values:
            return set()
        candidates = set(self.terms() and (p.doc_id for p in index.get_postings(self.values[0])))
        for doc_id in candidates.copy():
            positions = {p.doc_id: set(p.positions) for p in index.get_postings(self.values[0])}
            current = positions.get(doc_id, set())
            for term in self.values[1:]:
                posting = next((p for p in index.get_postings(term) if p.doc_id == doc_id), None)
                if posting is None or not any(pos + 1 in posting.positions for pos in current):
                    candidates.discard(doc_id)
                    break
                current = {pos + 1 for pos in current if pos + 1 in posting.positions}
        return candidates

    def terms(self) -> tuple[str, ...]:
        return self.values


@dataclass(frozen=True)
class And(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Or(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Not(Query):
    child: Query

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_meta) - self.child.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return self.child.terms()


TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|\S+')


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens = TOKEN_RE.findall(text)
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of query")
        self.pos += 1
        return token

    def parse(self) -> Query:
        if not self.tokens:
            raise ValueError("query is empty")
        result = self.or_expr()
        if self.peek() is not None:
            raise ValueError(f"unexpected token: {self.peek()}")
        return result

    def or_expr(self) -> Query:
        result = self.and_expr()
        while (self.peek() or "").upper() == "OR":
            self.take(); result = Or(result, self.and_expr())
        return result

    def and_expr(self) -> Query:
        result = self.not_expr()
        while True:
            token = self.peek()
            if token is None or token == ")" or token.upper() == "OR":
                return result
            if token.upper() == "AND":
                self.take()
            result = And(result, self.not_expr())

    def not_expr(self) -> Query:
        if (self.peek() or "").upper() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Query:
        token = self.take()
        if token == "(":
            result = self.or_expr()
            if self.take() != ")":
                raise ValueError("missing closing parenthesis")
            return result
        if token.startswith('"'):
            return Phrase(token[1:-1])
        return Term(token)


@functools.lru_cache(maxsize=256)
def parse(text: str) -> Query:
    return Parser(text).parse()
