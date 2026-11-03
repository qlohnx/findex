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
        values = (
            self.values
            if isinstance(self.values, tuple)
            else tuple(tokenize(self.values))
        )
        object.__setattr__(self, "values", values)

    def evaluate(self, index: Index) -> set[int]:
        values = self.terms()
        if not values:
            return set()
        if not index.positions:
            raise ValueError("phrase queries require an index built with --positions")
        candidates = {p.doc_id for p in index.get_postings(values[0])}
        for doc_id in candidates.copy():
            positions = {
                p.doc_id: set(p.positions) for p in index.get_postings(values[0])
            }
            current = positions.get(doc_id, set())
            for term in values[1:]:
                posting = next(
                    (p for p in index.get_postings(term) if p.doc_id == doc_id), None
                )
                if posting is None or not any(
                    pos + 1 in posting.positions for pos in current
                ):
                    candidates.discard(doc_id)
                    break
                current = {pos + 1 for pos in current if pos + 1 in posting.positions}
        return candidates

    def terms(self) -> tuple[str, ...]:
        return (
            tuple(tokenize(self.values))
            if isinstance(self.values, str)
            else self.values
        )


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


TOKEN_RE = re.compile(r'"[^"\n]*"|\(|\)|[^\s()"]+')


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens: list[str] = []
        end = 0
        for match in TOKEN_RE.finditer(text):
            if text[end : match.start()].strip():
                raise ValueError("invalid query syntax or unclosed quote")
            self.tokens.append(match.group())
            end = match.end()
        if text[end:].strip():
            raise ValueError("invalid query syntax or unclosed quote")
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
            self.take()
            result = Or(result, self.and_expr())
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
            if not tuple(tokenize(token[1:-1])):
                raise ValueError("phrase is empty")
            return Phrase(token[1:-1])
        if token == ")" or token.upper() in {"AND", "OR", "NOT"}:
            raise ValueError(f"unexpected token: {token}")
        if not tuple(tokenize(token)):
            raise ValueError(f"term contains no searchable tokens: {token}")
        return Term(token)


@functools.lru_cache(maxsize=256)
def parse(text: str) -> Query:
    return Parser(text).parse()
