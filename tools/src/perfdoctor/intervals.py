"""Interval arithmetic on half-open [start, end) intervals.

All functions take/return lists of (start, end) tuples. Inputs need not be sorted
or disjoint; outputs of `merge` are sorted and disjoint.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

Interval = Tuple[float, float]


def merge(intervals: Iterable[Interval]) -> List[Interval]:
    """Union of intervals -> sorted, disjoint list. Empty intervals are dropped."""
    xs = sorted((s, e) for s, e in intervals if e > s)
    out: List[Interval] = []
    for s, e in xs:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def total(intervals: Iterable[Interval]) -> float:
    """Total covered length (union, so overlaps are not double counted)."""
    return sum(e - s for s, e in merge(intervals))


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> List[Interval]:
    """Intersection of two interval sets (each is unioned first)."""
    a, b = merge(a), merge(b)
    i = j = 0
    out: List[Interval] = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_len(a: Iterable[Interval], b: Iterable[Interval]) -> float:
    return sum(e - s for s, e in intersect(a, b))


def complement(intervals: Iterable[Interval], window: Interval) -> List[Interval]:
    """Gaps inside `window` not covered by `intervals`."""
    ws, we = window
    out: List[Interval] = []
    cur = ws
    for s, e in merge(intervals):
        if e <= ws or s >= we:
            continue
        s, e = max(s, ws), min(e, we)
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < we:
        out.append((cur, we))
    return out


def concurrency_histogram(intervals: Iterable[Interval], window: Interval) -> dict:
    """Time spent with exactly k intervals active, for every k, inside `window`.

    This is the single most telling number for "is anything overlapping at all":
    if ~all of the time is spent at k<=1, no two kernels ever ran together.
    """
    ws, we = window
    events = []
    for s, e in intervals:
        s, e = max(s, ws), min(e, we)
        if e > s:
            events.append((s, 1))
            events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))  # ends before starts at equal t
    hist: dict = {}
    k = 0
    t = ws
    for x, d in events:
        if x > t:
            hist[k] = hist.get(k, 0.0) + (x - t)
            t = x
        k += d
    if we > t:
        hist[k] = hist.get(k, 0.0) + (we - t)
    return hist
