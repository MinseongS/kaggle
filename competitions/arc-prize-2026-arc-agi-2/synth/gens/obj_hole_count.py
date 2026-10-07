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


CONCEPT = "every object is a block with single-cell holes; recolour each object according to how many holes it has, using a per-task colour per hole count"


def block(rng, nh):
    for _ in range(100):
        h, w = rng.randint(3, 6), rng.randint(3, 7)
        inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(inner)
        holes = []
        for r, c in inner:
            if len(holes) == nh:
                break
            if all(abs(r - a) > 1 or abs(c - b) > 1 for a, b in holes):
                holes.append((r, c))
        if len(holes) == nh:
            return frozenset((r, c) for r in range(h) for c in range(w) if (r, c) not in holes)
    return None


def make_pair(rng, fg, pal, counts):
    H, W = rng.randint(12, 20), rng.randint(12, 20)
    occ = set()
    g = grid(H, W)
    o = grid(H, W)
    for nh in counts:
        s = block(rng, nh)
        if s is None:
            return None
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        for r, c in cells:
            g[r][c] = fg
            o[r][c] = pal[nh]
    return g, o


def generate(rng):
    cols = rng.sample(range(1, 10), 5)
    fg, pal = cols[0], cols[1:]
    n = rng.randint(3, 4) + 1
    while True:
        cl = [[rng.randint(0, 3) for _ in range(rng.randint(3, 5))] for _ in range(n)]
        seen = {x for c in cl[:-1] for x in c}
        if set(cl[-1]) <= seen and all(len(set(c)) >= 2 for c in cl):
            break
    pairs = []
    for counts in cl:
        while True:
            p = make_pair(rng, fg, pal, counts)
            if p:
                pairs.append({"input": p[0], "output": p[1]})
                break
    return {"train": pairs[:-1], "test": pairs[-1:]}
