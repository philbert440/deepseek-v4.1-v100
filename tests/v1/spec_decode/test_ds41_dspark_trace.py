# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_DSPARK_TRACE (debug, default off): per-step DSpark trace + scripts/dspark_trace_analyze.py."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode import ds41_dspark_trace as tr
from vllm.v1.spec_decode.dspark import DSparkProposer

ROOT = Path(__file__).resolve().parents[3]
K = 5
V = 16


def _analyzer():
    spec = importlib.util.spec_from_file_location("dspark_trace_analyze", ROOT / "scripts" / "dspark_trace_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_knob_off_no_trace(monkeypatch):
    monkeypatch.delenv(tr.TRACE_KNOB, raising=False)
    assert tr.trace_path() is None
    assert tr.DSparkTrace.for_runner(SimpleNamespace(speculative_config=object())) is None


def test_for_runner_names_file_per_stage(monkeypatch, tmp_path):
    monkeypatch.setenv(tr.TRACE_KNOB, str(tmp_path / "t.jsonl"))
    t = tr.DSparkTrace.for_runner(SimpleNamespace(speculative_config=object()))
    assert t is not None and t.path.endswith(".jsonl") and ".pp" in t.path
    assert tr.DSparkTrace.for_runner(SimpleNamespace(speculative_config=None)) is None


def test_collect_and_max_lines(tmp_path):
    t = tr.DSparkTrace(str(tmp_path / "t.jsonl"), max_lines=3)
    recs = t.collect(req_ids=["a", "b"], computed=[10, 20], sched={"a": 6, "b": 6},
                     spec_in={"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}, out=[[1, 2, 9], [11]],
                     ctx_qsl=[0, 6, 12], ctx_positions=list(range(10, 16)) + list(range(20, 26)),
                     anchor_pos=[13, 21], anchor_tok=[9, 11], drafts=[[1] * 5, [2] * 5], top5=None)
    assert recs[0]["ctx"] == [10, 15] and recs[1]["ctx"] == [20, 25]
    assert recs[0]["anchor_pos"] == 13 and recs[1]["out"] == [11] and recs[1]["drafts"] == [2] * 5
    t.record(recs)
    t.record(recs)
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["req"] == "a"


class _Model:
    vocab_size = V

    def markov_embed(self, token_ids):
        return token_ids.unsqueeze(-1).float()

    def markov_bias(self, embeds):
        # a bias that depends on the previous token, so later positions differ from pos0's argmax
        b = torch.zeros(embeds.shape[0], V)
        b[torch.arange(embeds.shape[0]), (embeds[:, 0].long() + 3) % V] = 0.5
        return b

    @staticmethod
    def map_draft_to_target(ids):
        return ids


def _proposer(trace_on: bool):
    p = object.__new__(DSparkProposer)
    p.num_speculative_tokens = K
    p.model = _Model()
    p._enable_probabilistic_draft_probs = False
    p._static_draft_vocab = None
    p.collect_confidence_logits = False
    p.confidence_scheduling_enabled = False
    p._max_verification_tokens = K
    p._ds41_pmin = 0.0
    p._ds41_max_rows = 0
    p._ds41_pmin_max_batch = 0
    p._ds41_trace_on = trace_on
    p.input_ids = torch.tensor([3, 0, 0, 0, 0, 7, 0, 0, 0, 0])
    p._anchor_indices = torch.arange(4) * K
    return p


def test_sample_draft_tokens_unchanged_with_trace():
    torch.manual_seed(0)
    logits = torch.randn(2 * K, V)
    hidden = torch.zeros(2 * K, 4)
    meta = SimpleNamespace(all_greedy=True)
    off, _ = _proposer(False)._sample_draft_tokens(hidden, meta, logits=logits.clone())
    p = _proposer(True)
    on, _ = p._sample_draft_tokens(hidden, meta, logits=logits.clone())
    assert torch.equal(off, on)
    ids, lps = p._ds41_trace_top5
    assert ids.shape == (2, 5) and lps.shape == (2, 5)
    # pos0 = anchor-dependent bias + base logits row 0 of each block; top-1 must be the pos0 draft
    assert ids[:, 0].tolist() == on.view(2, K)[:, 0].tolist()
    assert bool((lps <= 0).all()) and bool((lps[:, :-1] >= lps[:, 1:]).all())
    assert math.isclose(float(lps[0, 0].exp()), float(torch.softmax(
        logits.view(2, K, V)[0, 0] + p.model.markov_bias(p.model.markov_embed(torch.tensor([3])))[0], -1).max()),
        rel_tol=1e-5)


def test_on_step_writes_records(tmp_path):
    t = tr.DSparkTrace(str(tmp_path / "t.jsonl"))
    drafter = SimpleNamespace(num_query_per_req=K, positions=torch.tensor([13, 14, 15, 16, 17]),
                              input_ids=torch.tensor([9, 0, 0, 0, 0]), _dflash_num_context=6,
                              _context_positions_buffer=torch.arange(10, 16), _ds41_trace_top5=None)
    runner = SimpleNamespace(
        drafter=drafter,
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r"], num_computed_tokens_cpu=[10]),
        _draft_token_req_ids=["r"],
        _get_draft_token_ids_cpu=lambda: ([[4, 4, 4, 4, 4]], ["r"]),
    )
    so = SimpleNamespace(num_scheduled_tokens={"r": 6}, scheduled_spec_decode_tokens={"r": [1, 2, 3, 4, 5]})
    cad = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 6]))
    t.on_step(runner, so, [[1, 2, 9]], cad)
    rec = json.loads((tmp_path / "t.jsonl").read_text())
    assert rec == {"step": 0, "req": "r", "computed": 10, "sched": 6, "spec_in": [1, 2, 3, 4, 5], "out": [1, 2, 9],
                   "ctx": [10, 15], "anchor_pos": 13, "anchor_tok": 9, "drafts": [4, 4, 4, 4, 4], "top5": None,
                   "nonfinite": None}


def test_on_step_never_raises(tmp_path):
    t = tr.DSparkTrace(str(tmp_path / "t.jsonl"))
    t.on_step(SimpleNamespace(drafter=object(), input_batch=None), None, [], SimpleNamespace())
    assert t._failed


def _simulate(text: list[int], drafter, prompt_len: int = 8, k: int = K) -> list[dict]:
    """A greedy spec-decode run: the target emits ``text``; ``drafter(anchor_pos, step)`` -> k drafts."""
    recs = []
    step = 0
    tok_at = {prompt_len + i: t for i, t in enumerate(text)}       # token at absolute position
    # prefill: rows 0..prompt_len-1, samples text[0] at prompt_len
    ap = prompt_len
    d = drafter(ap, step, tok_at)
    recs.append({"step": step, "req": "r", "computed": 0, "sched": prompt_len, "spec_in": [], "out": [text[0]],
                 "ctx": [0, prompt_len - 1], "anchor_pos": ap, "anchor_tok": text[0], "drafts": d, "top5": None})
    computed = prompt_len           # row 0 of the next step = the bonus at position prompt_len
    while computed + k + 1 < prompt_len + len(text):
        step += 1
        spec = d
        out = []
        for i in range(k):
            if spec[i] == tok_at[computed + 1 + i]:
                out.append(spec[i])
            else:
                break
        out.append(tok_at[computed + 1 + len(out)])
        ap = computed + len(out)
        d = drafter(ap, step, tok_at)
        recs.append({"step": step, "req": "r", "computed": computed, "sched": k + 1, "spec_in": spec, "out": out,
                     "ctx": [computed, computed + k], "anchor_pos": ap, "anchor_tok": out[-1], "drafts": d,
                     "top5": [[tok_at.get(ap + 1, -1), -0.1], [1, -2.0], [2, -3.0], [3, -4.0], [5, -5.0]]})
        computed = ap
    return recs


def test_analyzer_healthy_pipeline():
    an = _analyzer()
    text = list(range(100, 300))

    def perfect(ap, step, tok_at):
        return [tok_at.get(ap + 1 + i, 0) for i in range(K)]

    res = an.analyze({"r": _simulate(text, perfect)})
    assert res["violations"] == {}
    assert res["pos0"] == 1.0 and res["shift"]["exact"] == res["shift"]["slots"]
    # every decode block whose pos0 truth is known (the prefill block carries no top-5) ranks it first
    assert res["top5_rank_of_truth_pos0"] == {"0": res["shift"]["blocks"] - 1}


def test_analyzer_detects_shift_and_alternation():
    an = _analyzer()
    text = list(range(100, 400))

    def shifted_after_accept(ap, step, tok_at):
        # the bug shape under test: every other block is off by one (predicts anchor + 2 ...)
        off = 1 if step % 2 else 0
        return [tok_at.get(ap + 1 + i + off, 0) for i in range(K)]

    res = an.analyze({"r": _simulate(text, shifted_after_accept)})
    assert res["violations"] == {}                      # bookkeeping is consistent: the drafts are just wrong
    assert 0.3 < res["pos0"] < 0.7
    rs = res["pos0_rejected_shift"]
    # every pos0-rejected block predicted anchor + 2 (the last block may run past the known text)
    assert rs["pos0_rejected"] > 10 and rs["d0_eq_truth1"] >= rs["pos0_rejected"] - 1
    assert res["shift"]["fwd"] > 0


def test_analyzer_detects_stale_drafts_and_bad_anchor():
    an = _analyzer()
    text = list(range(100, 200))

    def perfect(ap, step, tok_at):
        return [tok_at.get(ap + 1 + i, 0) for i in range(K)]

    recs = _simulate(text, perfect)
    recs[3]["spec_in"] = recs[1]["drafts"]               # step 3 verifies drafts proposed two steps earlier
    recs[4]["anchor_pos"] += 1
    recs[5]["anchor_tok"] = -7
    res = an.analyze({"r": recs})
    assert {"I6", "I3", "I4"} <= set(res["violations"])
    out = an.report(res)
    assert "VIOLATION I6" in out and "pos0 | previous step accepted" in out


def test_analyzer_cli(tmp_path, capsys):
    an = _analyzer()
    text = list(range(100, 160))
    recs = _simulate(text, lambda ap, s, tok_at: [tok_at.get(ap + 1 + i, 0) for i in range(K)])
    p = tmp_path / "t.pp1.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert an.main([str(p), "--json", str(tmp_path / "o.json")]) == 0
    assert "invariants I1-I6: all hold" in capsys.readouterr().out
    assert json.loads((tmp_path / "o.json").read_text())["pos0"] == 1.0


@pytest.mark.parametrize("bad", ["-1", "x"])
def test_trace_max_strict(monkeypatch, bad):
    monkeypatch.setenv(tr.TRACE_MAX_KNOB, bad)
    with pytest.raises(ValueError):
        tr._trace_max()


def test_analyzer_nan_draft_accounting():
    an = _analyzer()
    text = list(range(100, 260))

    def perfect(ap, step, tok_at):
        return [tok_at.get(ap + 1 + i, 0) for i in range(K)]

    recs = _simulate(text, perfect)
    # the live failure shape: the drafter emits NaN logits (drafts [0]*k) for the first steps, then recovers
    for r in recs[:6]:
        r["drafts"] = [0] * K
        r["top5"] = [[0, float("nan")]] * 5
        r["nonfinite"] = [0, 1, 2]
    for a, b in zip(recs, recs[1:]):
        if a["drafts"] == [0] * K and b["spec_in"]:
            b["spec_in"] = [0] * K
            b["out"] = b["out"][:1] if b["out"][0] != 0 else b["out"]
    res = an.analyze({"r": recs})
    assert res["nan_draft_steps"] == 6
    assert res["nonfinite_positions"] == {0: 6, 1: 6, 2: 6}
    assert res["finite_drafts"]["steps"] == res["decode_steps"] - 6 and res["finite_drafts"]["pos0"] == 1.0
    assert res["pos0"] < 1.0
    assert "NaN-draft steps 6" in an.report(res)
