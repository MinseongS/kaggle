import random

CONCEPT = ("one half of the grid is mirrored onto the other half (direction fixed per task), but only into empty "
           "cells; mirrored cells keep their colour or take a fixed task colour; existing cells are never overwritten")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    direction = rng.randint(0, 3)  # 0 L->R, 1 R->L, 2 T->B, 3 B->T
    fixed = rng.choice([None, rng.randint(1, 9)])
    pal = [c for c in range(1, 10) if c != fixed]

    def tf(m):
        if direction == 1:
            m = [r[::-1] for r in m]
        elif direction == 2:
            m = [list(x) for x in zip(*m)]
        elif direction == 3:
            m = [list(x) for x in zip(*m)][::-1]
        return [list(r) for r in m]

    def make():
        while True:
            h, hw = rng.randint(8, 16), rng.randint(4, 8)
            w = 2 * hw
            cols = rng.sample(pal, rng.randint(2, 3))
            dens = rng.uniform(0.35, 0.6)
            g = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(hw):
                    if rng.random() < dens:
                        g[r][c] = rng.choice(cols)
            keep = rng.uniform(0.2, 0.6)
            for r in range(h):
                for c in range(hw):
                    if g[r][c] and rng.random() < keep:
                        g[r][w - 1 - c] = g[r][c]
            # a few conflicting cells on the target side
            others = [c for c in pal if c not in cols]
            for _ in range(rng.randint(1, 4)):
                r, c = rng.randrange(h), rng.randint(hw, w - 1)
                g[r][c] = rng.choice(others)
            out = [row[:] for row in g]
            filled = 0
            for r in range(h):
                for c in range(hw, w):
                    src = g[r][w - 1 - c]
                    if out[r][c] == 0 and src:
                        out[r][c] = fixed if fixed else src
                        filled += 1
            if filled >= 3:
                return tf(g), tf(out)

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
