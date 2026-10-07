import random

DIRS4 = ((0, 1), (1, 0), (0, -1), (-1, 0))


def norm(cells):
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def rand_shape(rng, h, w, n):
    n = min(n, h * w)
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice(DIRS4)
        if 0 <= r + dr < h and 0 <= c + dc < w:
            cells.add((r + dr, c + dc))
    return norm(cells)


def dims(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def place(rng, occ, H, W, shape, margin=1, tries=300, r_lo=0, c_lo=0):
    sh, sw = dims(shape)
    if sh > H - r_lo or sw > W - c_lo:
        return None
    for _ in range(tries):
        r0, c0 = rng.randint(r_lo, H - sh), rng.randint(c_lo, W - sw)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if all((r + dr, c + dc) not in occ for r, c in cells
               for dr in range(-margin, margin + 1) for dc in range(-margin, margin + 1)):
            occ.update(cells)
            return cells
    return None


def grid(h, w, v=0):
    return [[v] * w for _ in range(h)]


CONCEPT = "several shapes are repeated (in arbitrary colours); the single shape that occurs exactly once is cropped out with its colour"


def make_pair(rng):
    H, W = rng.randint(10, 18), rng.randint(10, 18)
    nrep = rng.randint(1, 3)
    shapes = []
    while len(shapes) < nrep + 1:
        s = rand_shape(rng, rng.randint(2, 4), rng.randint(2, 4), rng.randint(3, 7))
        if s not in shapes:
            shapes.append(s)
    objs = [shapes[-1]] + [s for s in shapes[:-1] for _ in range(rng.randint(2, 3))]
    occ = set()
    g = grid(H, W)
    out = None
    for i, s in enumerate(objs):
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        col = rng.randint(1, 9)
        for r, c in cells:
            g[r][c] = col
        if i == 0:
            sh, sw = dims(s)
            out = grid(sh, sw)
            for r, c in s:
                out[r][c] = col
    return g, out


def generate(rng):
    while True:
        pairs = []
        n = rng.randint(3, 4) + 1
        while len(pairs) < n:
            p = make_pair(rng)
            if p:
                pairs.append({"input": p[0], "output": p[1]})
        if pairs[-1]["output"] not in [p["output"] for p in pairs[:-1]]:
            return {"train": pairs[:-1], "test": pairs[-1:]}
