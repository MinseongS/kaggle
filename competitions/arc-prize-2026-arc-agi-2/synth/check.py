"""Validate synthetic generators: uv run python synth/check.py synth/gens/a.py [more.py ...] [--n 200] [--show 1]"""
import argparse
import importlib.util
import json
import random
import sys


def load(path):
    spec = importlib.util.spec_from_file_location(path.rsplit("/", 1)[-1][:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert isinstance(getattr(m, "CONCEPT", None), str) and m.CONCEPT, "missing CONCEPT"
    return m


def check_grid(g, what):
    assert isinstance(g, list) and g and all(isinstance(r, list) for r in g), f"{what}: not a list of rows"
    w = len(g[0])
    assert 1 <= len(g) <= 30 and 1 <= w <= 30, f"{what}: size {len(g)}x{w}"
    assert all(len(r) == w for r in g), f"{what}: ragged"
    assert all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r), f"{what}: bad value"


def check_task(t, tag):
    assert set(t) == {"train", "test"}, f"{tag}: keys {set(t)}"
    assert 2 <= len(t["train"]) <= 5, f"{tag}: {len(t['train'])} train pairs"
    assert len(t["test"]) == 1, f"{tag}: {len(t['test'])} test pairs"
    for i, p in enumerate(t["train"] + t["test"]):
        check_grid(p["input"], f"{tag} pair{i} input"); check_grid(p["output"], f"{tag} pair{i} output")
        assert p["input"] != p["output"], f"{tag} pair{i}: input == output"
    ins = [json.dumps(p["input"]) for p in t["train"] + t["test"]]
    assert len(set(ins)) == len(ins), f"{tag}: duplicate inputs"
    outs = {json.dumps(p["output"]) for p in t["train"]}
    assert json.dumps(t["test"][0]["output"]) not in outs, f"{tag}: test output equals a train output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    bad = 0
    for path in a.paths:
        try:
            m = load(path)
            sizes = []
            for s in range(a.n):
                t = m.generate(random.Random(s))
                check_task(t, f"{path} seed{s}")
                assert t == m.generate(random.Random(s)), f"{path} seed{s}: not deterministic"
                sizes.append(sum(len(p["input"]) * len(p["input"][0]) for p in t["train"]) / len(t["train"]))
            uniq = len({json.dumps(m.generate(random.Random(s))) for s in range(min(a.n, 50))})
            print(f"OK   {path}: {a.n} tasks, mean input cells {sum(sizes)/len(sizes):.0f}, distinct(50) {uniq}  | {m.CONCEPT}")
            for s in range(a.show):
                t = m.generate(random.Random(s))
                for p in t["train"][:2] + t["test"]:
                    print("in:"); [print("  " + "".join(map(str, r))) for r in p["input"]]
                    print("out:"); [print("  " + "".join(map(str, r))) for r in p["output"]]
        except Exception as e:
            bad += 1
            print(f"FAIL {path}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
