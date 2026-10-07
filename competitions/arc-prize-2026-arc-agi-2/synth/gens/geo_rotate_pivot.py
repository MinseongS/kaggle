import random

CONCEPT = ("every object contains one pivot cell of a task colour; each object is rotated 90 degrees about its own "
           "pivot (clockwise or counter-clockwise, fixed per task)")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def _shape(rng, n):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        if abs(r + dr) <= 2 and abs(c + dc) <= 2:
            cells.add((r + dr, c + dc))
    return sorted(cells)


def generate(rng: random.Random) -> dict:
    pivot = rng.randint(1, 9)
    cw = rng.random() < 0.5
    pal = [c for c in range(1, 10) if c != pivot]

    def rot(rel, clockwise):
        return [(dc, -dr) if clockwise else (-dc, dr) for dr, dc in rel]

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            g = [[0] * w for _ in range(h)]
            out = [[0] * w for _ in range(h)]
            occ = set()
            placed, tries = 0, 0
            n_obj = rng.randint(2, 4)
            while placed < n_obj and tries < 300:
                tries += 1
                sh = _shape(rng, rng.randint(4, 7))
                p = rng.choice(sh)
                rel = [(r - p[0], c - p[1]) for r, c in sh]
                a = rot(rel, cw)
                b = rot(rel, not cw)
                if set(a) == set(rel) or set(a) == set(b):
                    continue
                pr, pc = rng.randrange(h), rng.randrange(w)
                both = [(pr + r, pc + c) for r, c in rel + a]
                if not all(0 <= r < h and 0 <= c < w for r, c in both):
                    continue
                halo = {(r + x, c + y) for r, c in both for x in (-1, 0, 1) for y in (-1, 0, 1)}
                if halo & occ:
                    continue
                occ |= set(both)
                col = rng.choice(pal)
                for r, c in rel:
                    g[pr + r][pc + c] = col
                for r, c in a:
                    out[pr + r][pc + c] = col
                g[pr][pc] = out[pr][pc] = pivot
                placed += 1
            if placed >= 2:
                return g, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
