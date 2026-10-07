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


CONCEPT = "crop every object and lay the crops side by side (row or column, per task) with one blank gap, ordered by cell count (ascending or descending, per task)"


def make_pair(rng, horiz, asc):
    H, W = rng.randint(10, 16), rng.randint(10, 16)
    k = rng.randint(3, 4)
    sizes = rng.sample(range(2, 11), k)
    cols = rng.sample(range(1, 10), k)
    occ = set()
    g = grid(H, W)
    objs = []
    for n, col in zip(sizes, cols):
        s = rand_shape(rng, rng.randint(2, 4), rng.randint(2, 4), n)
        if len(s) != n:
            return None
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        for r, c in cells:
            g[r][c] = col
        objs.append((n, s, col))
    objs.sort(key=lambda t: t[0], reverse=not asc)
    if horiz:
        oh = max(dims(s)[0] for _, s, _ in objs)
        ow = sum(dims(s)[1] for _, s, _ in objs) + k - 1
    else:
        ow = max(dims(s)[1] for _, s, _ in objs)
        oh = sum(dims(s)[0] for _, s, _ in objs) + k - 1
    o = grid(oh, ow)
    off = 0
    for _, s, col in objs:
        for r, c in s:
            if horiz:
                o[r][c + off] = col
            else:
                o[r + off][c] = col
        off += (dims(s)[1] if horiz else dims(s)[0]) + 1
    return g, o


def generate(rng):
    horiz, asc = rng.random() < 0.5, rng.random() < 0.5
    pairs = []
    n = rng.randint(3, 4) + 1
    while len(pairs) < n:
        p = make_pair(rng, horiz, asc)
        if p:
            pairs.append({"input": p[0], "output": p[1]})
    return {"train": pairs[:-1], "test": pairs[-1:]}
