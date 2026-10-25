#!/usr/bin/env python3
"""Analyze a VLLM_DS41_DSPARK_TRACE file (vllm/v1/spec_decode/ds41_dspark_trace.py).

Usage: dspark_trace_analyze.py TRACE.jsonl [--json OUT.json]

Invariants per request (a violation = a bookkeeping bug; the report prints the first few):
  I1 ctx[0] == computed                        drafter context starts at the step's row-0 position
  I2 out[:a] == spec_in[:a], out[a] != spec_in[a] (a = len(out) - 1)  greedy verify consistency
  I3 anchor_pos == computed + len(out)          drafter anchor = position of the bonus token (decode)
     anchor_pos == ctx[1] + 1                   (prefill: right after the last prompt position)
  I4 anchor_tok == out[-1]                      drafter anchor token = the step's bonus token
  I5 next.computed == computed + len(out)       no lost / duplicated steps
  I6 next.spec_in == drafts[:len(next.spec_in)] the next step verifies exactly the drafts proposed here (stale check)
Statistics: pos0 acceptance overall and conditioned on the previous step's accepted count, P(pos0 ok | previous pos0
ok / failed) (alternation), per-position conditional acceptance, an off-by-one test (draft[i] vs truth[i+1] /
truth[i-1]) and the rank of the true pos0 token in the drafter's top-5.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path: str) -> dict[str, list[dict]]:
    by_req: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_req[r["req"]].append(r)
    for recs in by_req.values():
        recs.sort(key=lambda r: r["step"])
    return by_req


def analyze(by_req: dict[str, list[dict]]) -> dict:
    viol: dict[str, list] = defaultdict(list)
    n_dec = 0
    pos0_ok = 0
    cond = defaultdict(lambda: [0, 0])           # prev accepted ('P' = prefill) -> [steps, pos0 ok]
    alt = {"prev_ok": [0, 0], "prev_fail": [0, 0]}
    per_pos = [[0, 0] for _ in range(16)]        # [reached, accepted] for position i given i-1 accepted
    shift = {"blocks": 0, "fwd": 0, "bwd": 0, "exact": 0, "slots": 0}
    rej_shift = {"pos0_rejected": 0, "d0_eq_truth1": 0, "d0_eq_anchor": 0}
    top5_rank = defaultdict(int)
    nan_drafts = 0                                   # steps whose drafter produced NaN logits (drafts [0]*k)
    fin = {"steps": 0, "pos0": 0, "tokens": 0}       # steps that verified FINITE drafts
    nonfinite_pos = defaultdict(int)
    for rid, recs in by_req.items():
        tok_at: dict[int, int] = {}
        for r in recs:
            out = r["out"]
            if not out:
                continue
            if r["spec_in"]:
                for j, t in enumerate(out):
                    tok_at[r["computed"] + 1 + j] = t
            elif r["ctx"] is not None:
                tok_at[r["ctx"][1] + 1] = out[-1]
        prev_acc = None
        prev_pos0 = None
        prev_nan = False
        for k, r in enumerate(recs):
            out, spec = r["out"], r["spec_in"]
            for p in r.get("nonfinite") or ():
                nonfinite_pos[p] += 1
            t5 = r.get("top5")
            is_nan = bool(t5) and isinstance(t5[0][1], float) and t5[0][1] != t5[0][1]
            nan_drafts += is_nan
            if spec and out and not prev_nan:
                fin["steps"] += 1
                fin["pos0"] += len(out) > 1
                fin["tokens"] += len(out)
            prev_nan = is_nan
            if not out:                               # partial prefill chunk: discarded sample
                prev_acc = None
                continue
            nxt = next((x for x in recs[k + 1:] if x["out"]), None)
            if r["ctx"] is not None and r["ctx"][0] != r["computed"]:
                viol["I1"].append((rid, r["step"], r["ctx"], r["computed"]))
            if spec:
                a = len(out) - 1
                if out[:a] != spec[:a] or (a < len(spec) and out[a] == spec[a]):
                    viol["I2"].append((rid, r["step"], spec, out))
                if r["anchor_pos"] is not None and r["anchor_pos"] != r["computed"] + len(out):
                    viol["I3"].append((rid, r["step"], r["anchor_pos"], r["computed"] + len(out)))
            elif r["ctx"] is not None and r["anchor_pos"] is not None and r["anchor_pos"] != r["ctx"][1] + 1:
                viol["I3"].append((rid, r["step"], r["anchor_pos"], r["ctx"][1] + 1))
            if r["anchor_tok"] is not None and r["anchor_tok"] != out[-1]:
                viol["I4"].append((rid, r["step"], r["anchor_tok"], out[-1]))
            if nxt is not None and spec and nxt["computed"] != r["computed"] + len(out):
                viol["I5"].append((rid, r["step"], nxt["computed"], r["computed"] + len(out)))
            if nxt is not None and nxt["spec_in"] and r["drafts"] is not None:
                if nxt["spec_in"] != r["drafts"][: len(nxt["spec_in"])]:
                    viol["I6"].append((rid, r["step"], r["drafts"], nxt["spec_in"]))
            if spec:
                n_dec += 1
                ok = len(out) > 1                      # draft 0 accepted
                pos0_ok += ok
                key = "P" if prev_acc is None else str(prev_acc)
                cond[key][0] += 1
                cond[key][1] += ok
                if prev_pos0 is not None:
                    b = alt["prev_ok" if prev_pos0 else "prev_fail"]
                    b[0] += 1
                    b[1] += ok
                prev_pos0 = ok
                a = len(out) - 1
                for i in range(len(spec)):
                    if i > a:
                        break
                    per_pos[i][0] += 1
                    per_pos[i][1] += i < a
                prev_acc = a
            else:
                prev_acc = None                      # prefill step: next step's pos0 is conditioned on 'P'
            # off-by-one test on the drafts proposed here, against the realized tokens
            d, ap = r["drafts"], r["anchor_pos"]
            if d and ap is not None:
                truth = [tok_at.get(ap + 1 + i) for i in range(len(d) + 1)]
                if truth[0] is not None:
                    shift["blocks"] += 1
                    for i in range(len(d)):
                        if truth[i] is None:
                            break
                        shift["slots"] += 1
                        shift["exact"] += d[i] == truth[i]
                        if truth[i + 1] is not None:
                            shift["fwd"] += d[i] == truth[i + 1]
                        prev_t = tok_at.get(ap + i)
                        if prev_t is not None:
                            shift["bwd"] += d[i] == prev_t
                    if d[0] != truth[0]:
                        rej_shift["pos0_rejected"] += 1
                        rej_shift["d0_eq_truth1"] += truth[1] is not None and d[0] == truth[1]
                        rej_shift["d0_eq_anchor"] += d[0] == tok_at.get(ap)
                    t5 = r.get("top5")
                    if t5:
                        ids = [int(x[0]) for x in t5]
                        top5_rank[str(ids.index(truth[0])) if truth[0] in ids else ">5"] += 1
    res = {
        "requests": len(by_req),
        "decode_steps": n_dec,
        "pos0": pos0_ok / max(n_dec, 1),
        "pos0_given_prev_accepted": {k: {"steps": v[0], "pos0": v[1] / max(v[0], 1)} for k, v in sorted(cond.items())},
        "pos0_given_prev_pos0": {k: {"steps": v[0], "pos0": v[1] / max(v[0], 1)} for k, v in alt.items()},
        "conditional_per_pos": [p[1] / p[0] for p in per_pos if p[0]],
        "shift": shift,
        "pos0_rejected_shift": rej_shift,
        "top5_rank_of_truth_pos0": dict(top5_rank),
        "nan_draft_steps": nan_drafts,
        "finite_drafts": {"steps": fin["steps"], "pos0": fin["pos0"] / max(fin["steps"], 1),
                          "AL": fin["tokens"] / max(fin["steps"], 1)},
        "nonfinite_positions": dict(sorted(nonfinite_pos.items())),
        "violations": {k: {"count": len(v), "first": v[:5]} for k, v in viol.items()},
    }
    return res


def report(res: dict) -> str:
    lines = [f"requests {res['requests']}  decode steps {res['decode_steps']}  pos0 {res['pos0']:.3f}"]
    lines.append("pos0 | previous step accepted: " + "  ".join(
        f"{k}:{v['pos0']:.2f}(n={v['steps']})" for k, v in res["pos0_given_prev_accepted"].items()))
    lines.append("pos0 | previous pos0: " + "  ".join(
        f"{k}:{v['pos0']:.2f}(n={v['steps']})" for k, v in res["pos0_given_prev_pos0"].items()))
    lines.append("conditional acceptance per position: " + " ".join(f"{x:.2f}" for x in res["conditional_per_pos"]))
    s = res["shift"]
    lines.append(f"draft slots {s['slots']}: exact {s['exact'] / max(s['slots'], 1):.3f}  "
                 f"draft[i]==truth[i+1] {s['fwd'] / max(s['slots'], 1):.3f}  "
                 f"draft[i]==truth[i-1] {s['bwd'] / max(s['slots'], 1):.3f}")
    r = res["pos0_rejected_shift"]
    lines.append(f"pos0-rejected blocks {r['pos0_rejected']}: d0==truth1 {r['d0_eq_truth1']}  d0==anchor {r['d0_eq_anchor']}")
    if res["top5_rank_of_truth_pos0"]:
        lines.append("rank of true pos0 token in drafter top-5: " + str(dict(sorted(res["top5_rank_of_truth_pos0"].items()))))
    f = res["finite_drafts"]
    lines.append(f"NaN-draft steps {res['nan_draft_steps']}; steps verifying finite drafts {f['steps']}: "
                 f"pos0 {f['pos0']:.3f} AL {f['AL']:.2f}")
    if res["nonfinite_positions"]:
        lines.append("non-finite tap rows zeroed at positions: " + str(res["nonfinite_positions"]))
    if res["violations"]:
        for k, v in sorted(res["violations"].items()):
            lines.append(f"VIOLATION {k}: {v['count']}  first: {v['first'][:3]}")
    else:
        lines.append("invariants I1-I6: all hold")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    res = analyze(load(argv[0]))
    print(report(res))
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
