# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""scripts/dspark_retune_judge.py: the judge table of scripts/dspark_retune_gpu.sh on synthetic arm outputs."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _judge():
    spec = importlib.util.spec_from_file_location("dspark_retune_judge", ROOT / "scripts" / "dspark_retune_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _acc(tok_s, pos0):
    return {c: {"tok_s": tok_s, "tokens": 600, "drafts": 200, "AL": 3.0, "per_draft": 0.4,
                "pos": {"0": pos0, "1": 0.5, "2": 0.4, "3": 0.3, "4": 0.2}, "finish": ["stop"]}
            for c in ("code", "structured", "math", "chat", "prose", "multilingual")}


def _b(tok_s, texts):
    return {c: {"tok_s": tok_s, "tokens": 600, "steps": 200, "accept": 0.4, "texts": texts, "finish": ["stop"]}
            for c in ("code", "prose", "structured", "chat")}


def test_judge_table(tmp_path):
    j = _judge()
    (tmp_path / "acc-p0.json").write_text(json.dumps(_acc(45.0, 0.75)))
    (tmp_path / "b1-p0.json").write_text(json.dumps(_b(44.0, ["a"])))
    (tmp_path / "b2-p0.json").write_text(json.dumps(_b(70.0, ["a", "b"])))
    (tmp_path / "sc-p0.txt").write_text("x\nSELF-CONSISTENCY PASS: worst mean 0.0001\n")
    (tmp_path / "acc-p075.json").write_text(json.dumps(_acc(50.0, 0.9)))
    (tmp_path / "b1-p075.json").write_text(json.dumps(_b(49.0, ["a"])))
    (tmp_path / "b2-p075.json").write_text(json.dumps(_b(60.0, ["a", "c"])))
    (tmp_path / "sc-p075.txt").write_text("SELF-CONSISTENCY FAIL: worst mean 0.9\n")
    (tmp_path / "acc-bad.json").write_text(json.dumps(_acc(40.0, 0.5)))
    (tmp_path / "b1-bad.json").write_text(json.dumps(_b(40.0, ["Z"])))
    out = j.judge(str(tmp_path), ["p0", "p075", "bad", "missing"])
    lines = out.splitlines()
    row = next(l for l in lines if l.startswith("p0 "))
    assert "  45.0" in row and " 3.00 " in row and "0.75 0.50 0.40 0.30 0.20" in row and "PASS" in row and "ref" in row
    row = next(l for l in lines if l.startswith("p075 "))
    assert "FAIL" in row and row.rstrip().endswith("same")
    row = next(l for l in lines if l.startswith("bad "))
    assert "DIFF:code,prose,structured,chat" in row and "n/a" in row
    assert any(l.startswith("missing") and "no acc json" in l for l in lines)
    b2 = [l for l in lines if l.startswith("p0 ") or l.startswith("p075 ")]
    assert len(b2) == 4 and "  70.0" in b2[2] and "  60.0" in b2[3]
    assert any("no b2 json" in l for l in lines)
