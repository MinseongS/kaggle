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


CONCEPT = "single cells of a per-task marker colour are scattered; keep only the (two-coloured) objects that a marker touches edge-to-edge, erase all other objects and all markers"


def make_pair(rng, mk):
    H, W = rng.randint(12, 18), rng.randint(12, 18)
    others = [c for c in range(1, 10) if c != mk]
    k = rng.randint(4, 6)
    occ = set()
    objs = []
    g = grid(H, W)
    for _ in range(k):
        s = rand_shape(rng, rng.randint(2, 4), rng.randint(2, 4), rng.randint(3, 8))
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        a, b = rng.sample(others, 2)
        for r, c in cells:
            g[r][c] = a if rng.random() < 0.6 else b
        objs.append(cells)
    owner = {cell: i for i, cells in enumerate(objs) for cell in cells}
    nt = rng.randint(1, k - 2)
    touched = rng.sample(range(k), nt)
    markers = set()

    def near(r, c):
        hit = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if q in markers:
                    hit.add("m")
                if q in owner:
                    hit.add(owner[q])
        return hit

    for i in touched:
        cand = sorted({(r + dr, c + dc) for r, c in objs[i] for dr, dc in DIRS4})
        cand = [(r, c) for r, c in cand if 0 <= r < H and 0 <= c < W and (r, c) not in owner and near(r, c) == {i}]
        if not cand:
            return None
        markers.add(rng.choice(cand))
    for _ in range(rng.randint(1, 4)):
        for _ in range(50):
            r, c = rng.randrange(H), rng.randrange(W)
            if (r, c) not in owner and not near(r, c):
                markers.add((r, c))
                break
    o = grid(H, W)
    for i in touched:
        for r, c in objs[i]:
            o[r][c] = g[r][c]
    for r, c in markers:
        g[r][c] = mk
    return g, o


def generate(rng):
    mk = rng.randint(1, 9)
    pairs = []
    n = rng.randint(3, 4) + 1
    while len(pairs) < n:
        p = make_pair(rng, mk)
        if p:
            pairs.append({"input": p[0], "output": p[1]})
    return {"train": pairs[:-1], "test": pairs[-1:]}
