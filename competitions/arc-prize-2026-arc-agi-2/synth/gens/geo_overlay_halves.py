import random

CONCEPT = ("a divider line splits the grid into two equal halves (side by side or stacked); overlay them cell by "
           "cell with a task-specific boolean rule (and/or/xor/nor/a-not-b) and output the half-size result in a task colour")

OPS = {
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "xor": lambda a, b: a != b,
    "nor": lambda a, b: not (a or b),
    "a_not_b": lambda a, b: a and not b,
    "b_not_a": lambda a, b: b and not a,
}


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    op = OPS[rng.choice(sorted(OPS))]
    ocol = rng.randint(1, 9)
    div = rng.choice([c for c in range(1, 10) if c != ocol])
    pal = [c for c in range(1, 10) if c not in (ocol, div)]

    def make():
        while True:
            hh, ww = rng.randint(4, 8), rng.randint(4, 8)
            ca, cb = rng.sample(pal, 2)
            A = [[rng.random() < 0.5 for _ in range(ww)] for _ in range(hh)]
            B = [[rng.random() < 0.5 for _ in range(ww)] for _ in range(hh)]
            combos = {(A[r][c], B[r][c]) for r in range(hh) for c in range(ww)}
            if len(combos) < 4:
                continue
            out = [[ocol if op(A[r][c], B[r][c]) else 0 for c in range(ww)] for r in range(hh)]
            ga = [[ca if v else 0 for v in row] for row in A]
            gb = [[cb if v else 0 for v in row] for row in B]
            if rng.random() < 0.5:
                g = [ga[r] + [div] + gb[r] for r in range(hh)]
            else:
                g = ga + [[div] * ww] + gb
            return g, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
