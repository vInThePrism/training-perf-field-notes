"""Why isn't my communication overlapping with compute?

Reads an Nsight Systems SQLite export (`nsys export --type sqlite`) and reports
four diagnostic observations for "overlap flags are on, but NCCL time is fully
exposed". The causes can coexist; these heuristics do not prove causality:

  H1  NCCL and compute kernels share a CUDA stream  -> serialised by construction
  H2  compute was *ready* (launched, and its stream predecessor finished) but did
      not start while NCCL ran                       -> event / dependency blocking
  H3  host was stuck in a synchronize call while NCCL ran -> no work submitted
  H4  NCCL kernels occupy most SMs                   -> no room to co-schedule

It also reports the time-weighted kernel concurrency (how often >=2 kernels ran
at once at all), the GPU-idle gap histogram (launch-bound vs a few big stalls),
and the arithmetic upper bound on what perfect overlap could save.

Usage:
    python -m perfdoctor.nsys_overlap trace.sqlite --device 0 [--num-sms 132]

The H2 metric deliberately uses the *strict* definition. The loose definition
("launched but not yet started") is satisfied by every kernel queued behind an
NCCL kernel on the same stream, so it is a consequence of H1, not an independent
cause; using it produces a circular argument.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import intervals as iv

NCCL_RE = re.compile(r"nccl", re.I)
SYNC_APIS = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
)
GAP_BUCKETS_US = [(0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, float("inf"))]


@dataclass
class Kernel:
    start: int
    end: int
    stream: int
    name: str
    launch_end: Optional[int] = None  # end of the cudaLaunchKernel call (host)
    grid: int = 0

    @property
    def is_nccl(self) -> bool:
        return bool(NCCL_RE.search(self.name))


@dataclass
class Report:
    device: int
    window_s: float
    busy_s: float
    idle_s: float
    nccl_s: float
    compute_s: float
    overlap_s: float
    per_stream: List[dict] = field(default_factory=list)
    concurrency: Dict[int, float] = field(default_factory=dict)
    mean_concurrency: float = 0.0
    gap_histogram: List[dict] = field(default_factory=list)
    h1_mixed_streams: List[int] = field(default_factory=list)
    h1_nccl_on_mixed_frac: float = 0.0
    h2_ready_waiting_during_nccl_frac: Optional[float] = None
    h2_idle_with_ready_work_frac: Optional[float] = None
    h3_sync_during_nccl_frac: Optional[float] = None
    h4_nccl_blocks_per_sm: List[dict] = field(default_factory=list)
    upper_bound_saving_s: float = 0.0
    verdict: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["concurrency"] = {str(k): v for k, v in sorted(self.concurrency.items())}
        return d


# --------------------------------------------------------------------------- io

def _columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def load_kernels(con: sqlite3.Connection, device: int) -> List[Kernel]:
    cols = _columns(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    grid = "gridX*gridY*gridZ" if {"gridX", "gridY", "gridZ"} <= set(cols) else "0"
    rt_join = ""
    rt_sel = "NULL"
    if _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        rt_join = (
            "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
            "ON r.correlationId = k.correlationId"
        )
        rt_sel = "r.end"
    q = f"""
        SELECT k.start, k.end, k.streamId, s.value, {rt_sel}, {grid}
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.{name_col}
        {rt_join}
        WHERE k.deviceId = ?
        ORDER BY k.start
    """
    return [Kernel(*row) for row in con.execute(q, (device,))]


def load_sync_calls(con: sqlite3.Connection, main_tid: Optional[int] = None) -> List[tuple]:
    """Host intervals spent inside synchronize APIs on the busiest launching thread."""
    if not _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    if main_tid is None:
        row = con.execute(
            """SELECT r.globalTid, COUNT(*) c FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN StringIds s ON s.id = r.nameId
               WHERE s.value LIKE 'cudaLaunchKernel%'
               GROUP BY r.globalTid ORDER BY c DESC LIMIT 1"""
        ).fetchone()
        if row is None:
            return []
        main_tid = row[0]
    marks = ",".join("?" * len(SYNC_APIS))
    q = f"""SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            WHERE r.globalTid = ? AND
                  (s.value IN ({marks}) OR substr(s.value,1,length(s.value)-6) IN ({marks}))"""
    # nsys sometimes suffixes API names with a version tag like "_v3020"; the second
    # IN clause strips a 6-char suffix so both spellings match.
    return [tuple(r) for r in con.execute(q, (main_tid, *SYNC_APIS, *SYNC_APIS))]


# --------------------------------------------------------------------- analysis

def analyse(kernels: List[Kernel], sync_calls: List[tuple] = (), num_sms: int = 132,
            device: int = 0) -> Report:
    if not kernels:
        raise ValueError("no kernels for this device in the trace")
    ns = 1e-9
    window = (min(k.start for k in kernels), max(k.end for k in kernels))
    all_iv = [(k.start, k.end) for k in kernels]
    nccl_iv = [(k.start, k.end) for k in kernels if k.is_nccl]
    comp_iv = [(k.start, k.end) for k in kernels if not k.is_nccl]

    busy = iv.total(all_iv)
    win = window[1] - window[0]
    nccl_t = iv.total(nccl_iv)
    comp_t = iv.total(comp_iv)
    overlap = iv.intersect_len(nccl_iv, comp_iv)

    r = Report(device=device, window_s=win * ns, busy_s=busy * ns,
               idle_s=(win - busy) * ns, nccl_s=nccl_t * ns, compute_s=comp_t * ns,
               overlap_s=overlap * ns)

    # ---- per stream / H1
    by_stream: Dict[int, List[Kernel]] = defaultdict(list)
    for k in kernels:
        by_stream[k.stream].append(k)
    nccl_on_mixed = 0.0
    for sid, ks in sorted(by_stream.items()):
        n = [(k.start, k.end) for k in ks if k.is_nccl]
        c = [(k.start, k.end) for k in ks if not k.is_nccl]
        mixed = bool(n) and bool(c)
        r.per_stream.append({
            "stream": sid, "nccl_s": iv.total(n) * ns, "compute_s": iv.total(c) * ns,
            "n_nccl": len(n), "n_compute": len(c), "mixed": mixed,
        })
        if mixed:
            r.h1_mixed_streams.append(sid)
            nccl_on_mixed += iv.total(n)
    r.h1_nccl_on_mixed_frac = nccl_on_mixed / nccl_t if nccl_t else 0.0

    # ---- concurrency
    hist = iv.concurrency_histogram(all_iv, window)
    r.concurrency = {k: v * ns for k, v in hist.items()}
    r.mean_concurrency = sum(k * v for k, v in hist.items()) / win if win else 0.0

    # ---- idle gaps
    gaps = iv.complement(all_iv, window)
    for lo, hi in GAP_BUCKETS_US:
        sel = [e - s for s, e in gaps if lo <= (e - s) / 1e3 < hi]
        r.gap_histogram.append({"bucket_us": f"{lo}-{hi}", "count": len(sel),
                                "total_s": sum(sel) * ns})

    # ---- H2 proxy: launched AND recorded stream predecessor done. Cross-stream
    # event dependencies and unrecorded work can still prevent actual readiness.
    have_launch = any(k.launch_end is not None for k in kernels)
    if have_launch and nccl_iv:
        waiting = []
        for sid, ks in by_stream.items():
            prev_end = None
            for k in sorted(ks, key=lambda x: x.start):
                if not k.is_nccl and k.launch_end is not None:
                    ready = max(k.launch_end, prev_end or k.launch_end)
                    if k.start > ready:
                        waiting.append((ready, k.start))
                prev_end = k.end if prev_end is None else max(prev_end, k.end)
        r.h2_ready_waiting_during_nccl_frac = (
            iv.intersect_len(waiting, nccl_iv) / nccl_t if nccl_t else 0.0)
        idle_t = iv.total(gaps)
        r.h2_idle_with_ready_work_frac = (
            iv.intersect_len(waiting, gaps) / idle_t if idle_t else 0.0)

    # ---- H3: host in sync while NCCL busy
    if sync_calls and nccl_iv:
        r.h3_sync_during_nccl_frac = iv.intersect_len(sync_calls, nccl_iv) / nccl_t

    # ---- H4: NCCL SM footprint
    agg: Dict[str, List[int]] = defaultdict(list)
    for k in kernels:
        if k.is_nccl and k.grid:
            agg[_short(k.name)].append(k.grid)
    for name, grids in sorted(agg.items(), key=lambda x: -len(x[1])):
        g = max(set(grids), key=grids.count)
        r.h4_nccl_blocks_per_sm.append({"kernel": name, "count": len(grids),
                                        "grid": g, "blocks_per_sm": g / num_sms})

    r.upper_bound_saving_s = (min(nccl_t, comp_t) - overlap) * ns
    r.verdict = _verdict(r)
    return r


def _short(name: str) -> str:
    return name.split("(")[0][:80]


def _verdict(r: Report) -> List[str]:
    v = []
    if r.nccl_s == 0:
        return ["No NCCL kernels in window; nothing to diagnose."]
    ov = r.overlap_s / r.nccl_s
    v.append(f"overlap = {ov:.2%} of NCCL time")
    if r.h1_nccl_on_mixed_frac > 0.5:
        v.append(
            f"H1 HOLDS for within-stream ordering: {r.h1_nccl_on_mixed_frac:.1%} of NCCL "
            f"time sits on streams that also run compute {r.h1_mixed_streams}. Work "
            "within each stream is ordered; compute on other streams may still overlap. "
            "Inspect communication stream placement and CUDA_DEVICE_MAX_CONNECTIONS "
            "in the actual framework and device configuration.")
    else:
        v.append("H1 observation: NCCL mostly runs on streams without recorded compute.")
    if r.h2_ready_waiting_during_nccl_frac is not None:
        v.append(
            "H2 queue-wait proxy: launched compute whose recorded stream predecessor "
            "finished covers "
            f"{r.h2_ready_waiting_during_nccl_frac:.1%} of NCCL time; only "
            f"{r.h2_idle_with_ready_work_frac:.1%} of kernel-idle time had such work "
            "waiting. Cross-stream events and unrecorded activity are not resolved.")
    if r.h3_sync_during_nccl_frac is not None:
        v.append("H3 observation: selected launching thread was inside a synchronize call for "
                 f"{r.h3_sync_during_nccl_frac:.1%} of NCCL time.")
    if r.h4_nccl_blocks_per_sm:
        worst = max(x["blocks_per_sm"] for x in r.h4_nccl_blocks_per_sm)
        v.append(f"H4 grid-size proxy: largest modal NCCL grid has {worst:.2f} blocks/SM. "
                 "Grid size does not measure resident occupancy, free SMs, or resource contention.")
    low = sum(t for k, t in r.concurrency.items() if k <= 1)
    v.append(f"mean kernel concurrency {r.mean_concurrency:.3f}; "
             f"{low / r.window_s:.1%} of the window had <=1 kernel running.")
    v.append(f"upper bound from perfect overlap: {r.upper_bound_saving_s:.4g} s "
             f"({r.upper_bound_saving_s / r.window_s:.1%} of window) -- a ceiling, "
             "not an expected gain (synchronous collectives such as MoE all-to-all "
             "have true data dependencies).")
    return v


def print_report(r: Report) -> None:
    print(f"== device {r.device}: window {r.window_s:.3f}s  busy {r.busy_s:.3f}s  "
          f"kernel-idle {r.idle_s:.3f}s ({r.idle_s / r.window_s:.1%})")
    print(f"   NCCL {r.nccl_s:.3f}s  compute {r.compute_s:.3f}s  overlap {r.overlap_s:.4f}s")
    print("\n-- per stream")
    for s in r.per_stream:
        flag = "  <== MIXED" if s["mixed"] else ""
        print(f"   stream {s['stream']:>5}: NCCL {s['nccl_s']:8.3f}s ({s['n_nccl']:>6})"
              f"  compute {s['compute_s']:8.3f}s ({s['n_compute']:>7}){flag}")
    print("\n-- concurrency (time with exactly k kernels running)")
    for k, t in sorted(r.concurrency.items()):
        print(f"   k={k}: {t:9.3f}s  {t / r.window_s:6.2%}")
    print("\n-- gaps without recorded kernels (copies/memsets are not included)")
    for g in r.gap_histogram:
        print(f"   {g['bucket_us']:>12} us: {g['count']:>8}  {g['total_s']:9.3f}s")
    print("\n-- verdict")
    for line in r.verdict:
        print("   *", line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-sms", type=int, default=132, help="H100/H800 SXM: 132")
    ap.add_argument("--json", help="also write the report as JSON here")
    a = ap.parse_args(argv)
    con = sqlite3.connect(a.sqlite)
    rep = analyse(load_kernels(con, a.device), load_sync_calls(con), a.num_sms, a.device)
    print_report(rep)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
