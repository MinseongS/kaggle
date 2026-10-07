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


CONCEPT = "count the objects (not cells) of each colour; output a bar chart with one bar per colour whose length is that count, bars sorted longest first (vertical bottom-up or horizontal left-right, per task)"


def make_pair(rng, vertical):
    H, W = rng.randint(13, 20), rng.randint(13, 20)
    k = rng.randint(2, 4)
    cols = rng.sample(range(1, 10), k)
    counts = rng.sample(range(1, 6), k)
    occ = set()
    g = grid(H, W)
    for col, cnt in zip(cols, counts):
        for _ in range(cnt):
            s = rand_shape(rng, rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 5))
            cells = place(rng, occ, H, W, s)
            if cells is None:
                return None
            for r, c in cells:
                g[r][c] = col
    bars = sorted(zip(counts, cols), reverse=True)
    m = bars[0][0]
    if vertical:
        o = grid(m, k)
        for j, (cnt, col) in enumerate(bars):
            for r in range(m - cnt, m):
                o[r][j] = col
    else:
        o = grid(k, m)
        for j, (cnt, col) in enumerate(bars):
            for c in range(cnt):
                o[j][c] = col
    return g, o


def generate(rng):
    vertical = rng.random() < 0.5
    while True:
        pairs = []
        n = rng.randint(3, 4) + 1
        while len(pairs) < n:
            p = make_pair(rng, vertical)
            if p:
                pairs.append({"input": p[0], "output": p[1]})
        if pairs[-1]["output"] not in [p["output"] for p in pairs[:-1]]:
            return {"train": pairs[:-1], "test": pairs[-1:]}
