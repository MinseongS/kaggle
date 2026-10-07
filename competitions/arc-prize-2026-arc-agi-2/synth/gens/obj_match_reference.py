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


CONCEPT = "a framed reference shape sits in the top-left corner; objects with exactly that shape take the reference colour, all other objects are erased"


def make_pair(rng, frame):
    H, W = rng.randint(12, 18), rng.randint(12, 18)
    ref = rand_shape(rng, 3, 3, rng.randint(3, 6))
    rh, rw = dims(ref)
    rc, xc = rng.sample([c for c in range(1, 10) if c != frame], 2)
    g = grid(H, W)
    o = grid(H, W)
    occ = set()
    for r in range(rh + 2):
        for c in range(rw + 2):
            occ.add((r, c))
            v = frame if r in (0, rh + 1) or c in (0, rw + 1) else 0
            g[r][c] = o[r][c] = v
    for r, c in ref:
        g[r + 1][c + 1] = o[r + 1][c + 1] = rc
    nm, nd = rng.randint(1, 3), rng.randint(2, 4)
    objs = [ref] * nm
    while len(objs) < nm + nd:
        if rng.random() < 0.4:  # rotated / mirrored reference as a near-miss
            d = norm([(c, -r) for r, c in ref]) if rng.random() < 0.5 else norm([(r, -c) for r, c in ref])
        else:
            d = rand_shape(rng, 3, 3, rng.randint(3, 6))
        if d != ref:
            objs.append(d)
    rng.shuffle(objs)
    for s in objs:
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        for r, c in cells:
            g[r][c] = xc
            if s == ref:
                o[r][c] = rc
    return g, o


def generate(rng):
    frame = rng.randint(1, 9)
    pairs = []
    n = rng.randint(3, 4) + 1
    while len(pairs) < n:
        p = make_pair(rng, frame)
        if p:
            pairs.append({"input": p[0], "output": p[1]})
    return {"train": pairs[:-1], "test": pairs[-1:]}
