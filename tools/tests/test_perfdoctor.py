import math
import sqlite3

import pytest

from perfdoctor import intervals as iv
from perfdoctor import e2e_mfu, amdahl, silent_checks as sc
from perfdoctor.nsys_overlap import analyse, load_kernels, load_sync_calls


# ------------------------------------------------------------------ intervals

def test_interval_known_answers():
    assert iv.intersect_len([(0, 10), (20, 30)], [(5, 25)]) == 10
    assert iv.intersect_len([(0, 1)], [(2, 3)]) == 0
    assert iv.intersect_len([(0, 10)], [(2, 3)]) == 1
    assert sum(e - s for s, e in iv.complement([(10, 20), (30, 40)], (0, 50))) == 30
    assert iv.total([(0, 10), (5, 15), (20, 25)]) == 20


def test_concurrency_histogram():
    h = iv.concurrency_histogram([(0, 10), (5, 15)], (0, 20))
    assert h == {1: 10, 2: 5, 0: 5}


# ------------------------------------------------------------------ nsys

def _make_trace(path, same_stream: bool):
    """Tiny synthetic nsys export: alternating NCCL/compute on one device."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, deviceId INT,
            streamId INT, correlationId INT, demangledName INT, shortName INT,
            gridX INT, gridY INT, gridZ INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, globalTid INT,
            correlationId INT, nameId INT);
    """)
    con.executemany("INSERT INTO StringIds VALUES(?,?)", [
        (1, "ncclDevKernel_AllGather_RING_LL(ncclDevComm*)"),
        (2, "nvjet_gemm_kernel"),
        (3, "cudaLaunchKernel_v7000"),
        (4, "cudaStreamSynchronize_v3020"),
    ])
    us = 1000
    rows, rts = [], []
    t, cid = 0, 0
    for i in range(10):
        cid += 1  # NCCL, 1 block
        rows.append((t, t + 100 * us, 0, 7, cid, 1, 1, 1, 1, 1))
        rts.append((t - 5 * us, t - 4 * us, 42, cid, 3))
        cid += 1  # compute queued behind it, launched at the same time
        cs = 20 if not same_stream else 7
        start = t + 100 * us if same_stream else t + 10 * us
        rows.append((start, start + 40 * us, 0, cs, cid, 2, 2, 264, 1, 1))
        rts.append((t - 3 * us, t - 2 * us, 42, cid, 3))
        t += 200 * us
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", rts)
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)",
                (0, 50 * us, 42, 999, 4))
    con.commit()
    return con


def test_h1_detected_when_same_stream(tmp_path):
    con = _make_trace(tmp_path / "a.sqlite", same_stream=True)
    r = analyse(load_kernels(con, 0), load_sync_calls(con))
    assert r.h1_mixed_streams == [7]
    assert r.h1_nccl_on_mixed_frac == pytest.approx(1.0)
    assert r.overlap_s == 0
    # compute was launched long before it started, but its stream predecessor
    # (the NCCL kernel) had not finished -> strict H2 must NOT fire
    assert r.h2_ready_waiting_during_nccl_frac == pytest.approx(0.0)
    assert r.h3_sync_during_nccl_frac == pytest.approx(0.05)  # 50us of 1000us
    assert r.h4_nccl_blocks_per_sm[0]["grid"] == 1
    assert any("H1 HOLDS" in v for v in r.verdict)
    assert max(r.concurrency) == 1


def test_overlap_when_separate_streams(tmp_path):
    con = _make_trace(tmp_path / "b.sqlite", same_stream=False)
    r = analyse(load_kernels(con, 0))
    assert r.h1_mixed_streams == []
    assert r.overlap_s == pytest.approx(10 * 40e-6)
    assert r.concurrency[2] == pytest.approx(10 * 40e-6)


# ------------------------------------------------------------------ e2e

def test_e2e_reduces_to_share_of_wallclock():
    x = e2e_mfu.E2EInputs(mfu_step=0.5, t_step=10, n_full_steps=10, t_init=50, t_final=50)
    r = e2e_mfu.estimate(x)
    assert r.training_e2e_mfu == pytest.approx(0.5 * 100 / 200)


def test_e2e_worked_example_matches_note():
    full, part = e2e_mfu.steps_for_packed_epochs(952, 64, 3)
    assert (full, part) == (42, [0.875] * 3)
    r = e2e_mfu.estimate(e2e_mfu.E2EInputs(
        mfu_step=0.4431, t_step=113.194, n_full_steps=full, partial_step_fractions=part,
        t_init=260.714, eval_times=[26.146] * 55, t_ckpt_exposed=19.5, t_final=181.123))
    assert r.training_e2e_mfu == pytest.approx(0.3220, abs=5e-4)


def test_workload_e2e_counts_eval_flops():
    r = e2e_mfu.estimate(e2e_mfu.E2EInputs(
        mfu_step=0.4, t_step=1, n_full_steps=10, eval_times=[10], eval_mfus=[0.6]))
    assert r.workload_e2e_mfu == pytest.approx((4 + 6) / 20)
    assert r.training_e2e_mfu == pytest.approx(4 / 20)


# ------------------------------------------------------------------ amdahl

def test_amdahl_same_saving_smaller_gain():
    a = amdahl.Config("a", 10, 9, 1000, 0.4)
    b = amdahl.Config("b", 100, 99, 1000, 0.4)
    assert a.delta_mfu_pts > 5 * b.delta_mfu_pts


# ------------------------------------------------------------------ silent checks

def test_backend_from_log_catches_silent_fallback():
    log = "FlashAttention-3 is not installed.\nUsing torch SDPA for faster training"
    with pytest.raises(AssertionError):
        sc.assert_backend(log, "fa3")
    sc.assert_backend("Using FlashAttention-3 ...", "fa3")


def test_overlap_env_warning():
    assert sc.overlap_env_warnings(1, True, {"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    assert not sc.overlap_env_warnings(2, True, {"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    assert not sc.overlap_env_warnings(1, True, {"CUDA_DEVICE_MAX_CONNECTIONS": "8"})


def test_kernel_slots_keep_incumbent_when_candidate_ineligible():
    s = sc.KernelSlots()
    s.claim("loss", "liger_flce", eligible=True)
    assert s.claim("loss", "cce", eligible=False, reason="VL+ZeRO3") == "liger_flce"
    assert s.table() == {"loss": "liger_flce"}
    assert s.claim("loss", "cce", eligible=True) == "cce"


def test_optimizer_orphaned_after_param_swap():
    torch = pytest.importorskip("torch")
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW(m.parameters())
    assert sc.optimizer_coverage(m, opt).ok
    # emulate a conversion that rebuilds Parameters after the optimizer exists
    m[0].weight = torch.nn.Parameter(m[0].weight.detach().clone())
    rep = sc.optimizer_coverage(m, opt)
    assert not rep.ok
    assert rep.missing_from_optimizer == ["0.weight"] and rep.orphaned_in_optimizer == 1


def test_muon_routing_uses_logical_shape_under_zero3():
    torch = pytest.importorskip("torch")
    w = torch.nn.Parameter(torch.zeros(16))          # a ZeRO-3 1-D shard
    w.ds_shape = torch.Size([64, 64])                # its real shape
    gate_proj = torch.nn.Parameter(torch.zeros(8)); gate_proj.ds_shape = torch.Size([8, 8])
    router = torch.nn.Parameter(torch.zeros(4)); router.ds_shape = torch.Size([2, 8])
    out = sc.muon_eligible([("layers.0.attn.q_proj.weight", w),
                            ("layers.0.mlp.gate_proj.weight", gate_proj),
                            ("layers.0.mlp.gate.weight", router)])
    assert out["muon"] == ["layers.0.attn.q_proj.weight", "layers.0.mlp.gate_proj.weight"]
    assert out["ndim_router_muon"] == []   # the bug: ndim sees shards -> 0 Muon params
