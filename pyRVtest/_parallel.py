"""Opt-in threaded execution of the per-market loops on the solve path.

The per-market loops in the analytical (nested-)logit backends and in the
demand-adjustment stage are embarrassingly parallel: each market is computed
independently and writes a *disjoint* output region. Threads are used rather
than processes so the (potentially multi-GB) Jacobian cache is shared read-only
with zero serialization.

The public knob is ``Problem.solve(..., n_jobs=1)``. ``n_jobs=1`` (the default)
runs the exact serial code path, so results are unchanged. Because every market
writes to disjoint memory, the output is **bit-identical regardless of**
``n_jobs`` or worker scheduling -- threading never changes correctness.

Whether threading *speeds things up* depends on the workload and runtime. The
per-market math (``np.linalg.solve`` / ``einsum``) releases the GIL, but only
its large-array portion does; for small per-market blocks (few products per
market, the typical scanner-data case) per-iteration Python overhead dominates,
the loop is GIL-bound, and ``n_jobs > 1`` can be *slower* than serial. Real
speedups need either large per-market blocks (many products per market) or a
free-threaded (no-GIL) CPython build. Benchmark before enabling.

Helpers
-------
resolve_n_jobs
    Normalize / validate a user-supplied ``n_jobs`` to a positive worker count.
for_each
    Run ``worker(item)`` for every item, serially or across a thread pool.
    The ``worker`` MUST write only to disjoint outputs (no shared accumulation,
    no lazy-cache first-touch -- warm such caches in the main thread first).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

T = TypeVar('T')


def resolve_n_jobs(n_jobs: object) -> int:
    """Normalize ``n_jobs`` to a positive worker count.

    ``None`` or ``1`` -> ``1`` (serial). ``-1`` -> ``os.cpu_count()`` (all
    cores, at least 1). Any other value ``< 1`` raises ``ValueError``.
    """
    if n_jobs is None:
        return 1
    try:
        n = int(n_jobs)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise ValueError(
            f"Expected n_jobs to be an integer (>= 1, or -1 for all cores). "
            f"Received {n_jobs!r}."
        )
    if n == -1:
        return max(1, os.cpu_count() or 1)
    if n < 1:
        raise ValueError(
            f"Expected n_jobs >= 1, or -1 for all cores. "
            f"Received {n}. "
            f"Fix: pass n_jobs=1 (serial), a positive worker count, or -1."
        )
    return n


def _chunk(items: Sequence[T], n_chunks: int) -> List[List[T]]:
    """Split ``items`` into at most ``n_chunks`` contiguous, near-equal lists."""
    n = len(items)
    n_chunks = max(1, min(n_chunks, n))
    base, extra = divmod(n, n_chunks)
    chunks: List[List[T]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(list(items[start:start + size]))
        start += size
    return chunks


def for_each(
        items: Sequence[T], worker: Callable[[T], object], n_jobs: int
) -> None:
    """Call ``worker(item)`` for every item, serially or across threads.

    Serial (identical to a plain ``for`` loop) when ``n_jobs <= 1`` or there is
    at most one item. Otherwise the items are split into ``n_jobs * 4``
    contiguous chunks (to amortize dispatch overhead) and run on a
    ``ThreadPoolExecutor``. ``worker`` must write only to disjoint outputs.
    """
    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        for item in items:
            worker(item)
        return

    chunks = _chunk(items, n_jobs * 4)

    def _run_chunk(chunk: List[T]) -> None:
        for item in chunk:
            worker(item)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # list() forces consumption so any worker exception propagates here.
        list(executor.map(_run_chunk, chunks))
