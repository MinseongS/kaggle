import random

CONCEPT = ("each seed cell has a tail cell (task colour) diagonally behind it; a diagonal line of the seed colour "
           "runs away from the tail, bouncing off the two side walls, until it reaches the far wall (axis fixed per task)")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    tail = rng.randint(1, 9)
    transpose = rng.random() < 0.5
    pal = [c for c in range(1, 10) if c != tail]

    def path_of(r, c, dr, dc, h, w):
        p = []
        bounced = False
        while not (r == 0 and dr < 0) and not (r == h - 1 and dr > 0):
            if not 0 <= c + dc < w:
                dc = -dc
                bounced = True
            r, c = r + dr, c + dc
            p.append((r, c))
        return p, bounced

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(5, 12)
            g = [[0] * w for _ in range(h)]
            seeds, fixed_cells = [], set()
            for _ in range(rng.randint(1, 3)):
                dr, dc = rng.choice([-1, 1]), rng.choice([-1, 1])
                r, c = rng.randint(1, h - 2), rng.randrange(w)
                tr, tc = r - dr, c - dc
                if not 0 <= tc < w:
                    continue
                if {(r, c), (tr, tc)} & {(a + x, b + y) for a, b in fixed_cells for x in (-1, 0, 1) for y in (-1, 0, 1)}:
                    continue
                fixed_cells |= {(r, c), (tr, tc)}
                seeds.append((r, c, dr, dc, tr, tc, rng.choice(pal)))
            if not seeds:
                continue
            out = [[0] * w for _ in range(h)]
            used, ok, any_bounce = set(), True, False
            for r, c, dr, dc, tr, tc, col in seeds:
                p, b = path_of(r, c, dr, dc, h, w)
                any_bounce |= b
                if len(p) < 3 or set(p) & (used | fixed_cells):
                    ok = False
                    break
                used |= set(p)
                for a, bb in p:
                    out[a][bb] = col
            if not ok or not any_bounce:
                continue
            for r, c, dr, dc, tr, tc, col in seeds:
                g[r][c] = out[r][c] = col
                g[tr][tc] = out[tr][tc] = tail
            if transpose:
                g = [list(x) for x in zip(*g)]
                out = [list(x) for x in zip(*out)]
            return g, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
