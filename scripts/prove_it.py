"""Lab 02 «Prove it to yourself»: п'ять експериментів з теорії.

uv run python scripts/prove_it.py
"""

import subprocess
import sys
import time
import tracemalloc
from array import array
from dataclasses import dataclass


def exp1_resize() -> None:
    print("1) getsizeof(dict) після n вставок:")
    d: dict[int, int] = {}
    last = sys.getsizeof(d)
    print(f"   n=0  {last} B")
    for n in range(1, 40):
        d[n] = n
        size = sys.getsizeof(d)
        if size != last:
            print(f"   n={n:<2} {size} B   <- resize")
            last = size


def exp2_bad_hash() -> None:
    print("2) __hash__ = 42: усе колізує, `in` стає O(n)")

    class Good:
        def __init__(self, v: int) -> None:
            self.v = v

        def __eq__(self, o: object) -> bool:
            return isinstance(o, Good) and self.v == o.v

        __hash__ = None  # type: ignore[assignment]  # те, що робить Python сам

    try:
        {Good(1)}
    except TypeError as e:
        print(f"   без __hash__: TypeError: {e}")

    class Const(Good):
        def __hash__(self) -> int:
            return 42

    class Fine(Good):
        def __hash__(self) -> int:
            return hash(self.v)

    for cls in (Fine, Const):
        s = {cls(i) for i in range(10_000)}
        probe = [cls(i) for i in range(0, 10_000, 10)]
        t0 = time.perf_counter()
        for p in probe:
            assert p in s
        print(f"   {cls.__name__:<6} 1000 lookups: {time.perf_counter() - t0:.4f} s")


def exp3_hash_seed() -> None:
    print("3) hash('hello') у двох процесах / з PYTHONHASHSEED=0:")
    cmd = [sys.executable, "-c", "print(hash('hello'))"]
    for env in ({}, {}, {"PYTHONHASHSEED": "0"}, {"PYTHONHASHSEED": "0"}):
        out = subprocess.run(
            cmd, capture_output=True, text=True, env=env or None, check=True
        )
        print(f"   {env or 'random'!s:<28} {out.stdout.strip()}")


def exp4_slots() -> None:
    print("4) 1 000 000 екземплярів з 3 полями (tracemalloc):")

    @dataclass
    class Plain:
        a: int
        b: int
        c: int

    @dataclass(slots=True)
    class Slotted:
        a: int
        b: int
        c: int

    peaks = {}
    for cls in (Plain, Slotted):
        tracemalloc.start()
        objs = [cls(i, i, i) for i in range(1_000_000)]
        _, peaks[cls.__name__] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del objs
    for name, peak in peaks.items():
        print(f"   {name:<8} {peak / 1024 / 1024:7.1f} MB")
    print(f"   ratio: {peaks['Plain'] / peaks['Slotted']:.2f}x")


def exp5_array() -> None:
    print("5) array('I') vs list для 10**6 чисел:")
    n = 10**6
    a = array("I", range(n))
    lst = list(range(n))
    print(f"   getsizeof(array) = {sys.getsizeof(a) / 1e6:.1f} MB")
    print(
        f"   getsizeof(list)  = {sys.getsizeof(lst) / 1e6:.1f} MB  (лише масив вказівників)"
    )
    ints = sum(sys.getsizeof(x) for x in lst)
    print(
        f"   + самі int-об'єкти: {ints / 1e6:.1f} MB -> разом {(sys.getsizeof(lst) + ints) / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    exp1_resize()
    exp2_bad_hash()
    exp3_hash_seed()
    exp4_slots()
    exp5_array()
