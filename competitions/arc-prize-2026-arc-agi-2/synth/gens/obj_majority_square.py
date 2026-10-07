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


CONCEPT = "find the colour that forms the most separate objects (not the most cells); output an NxN square of that colour where N is its object count"


def make_pair(rng):
    H, W = rng.randint(12, 20), rng.randint(12, 20)
    k = rng.randint(3, 4)
    cols = rng.sample(range(1, 10), k)
    counts = [rng.randint(1, 4) for _ in range(k)]
    counts[0] = max(counts) + 1
    if counts[0] < 2 or counts[0] > 5:
        return None
    occ = set()
    g = grid(H, W)
    cells_of = {}
    for i, (col, cnt) in enumerate(zip(cols, counts)):
        for _ in range(cnt):
            if i == 0:
                s = rand_shape(rng, 2, 2, rng.randint(1, 3))
            else:
                s = rand_shape(rng, rng.randint(2, 4), rng.randint(2, 4), rng.randint(3, 9))
            cells = place(rng, occ, H, W, s)
            if cells is None:
                return None
            for r, c in cells:
                g[r][c] = col
            cells_of[col] = cells_of.get(col, 0) + len(cells)
    if max(cells_of.values()) <= cells_of[cols[0]] or sorted(cells_of.values())[-1] == sorted(cells_of.values())[-2]:
        return None  # the cell-count majority must be a different (unique) colour
    n = counts[0]
    return g, grid(n, n, cols[0])


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
