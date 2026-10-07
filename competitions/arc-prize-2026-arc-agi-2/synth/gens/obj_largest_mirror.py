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


CONCEPT = "among several two-coloured objects, crop the one with the most cells and mirror it (left-right or up-down, fixed per task)"


def make_pair(rng, axis):
    H, W = rng.randint(10, 18), rng.randint(10, 18)
    k = rng.randint(3, 5)
    sizes = sorted(rng.sample(range(4, 16), k), reverse=True)
    occ = set()
    g = grid(H, W)
    crop = None
    for i, n in enumerate(sizes):
        s = rand_shape(rng, rng.randint(3, 5), rng.randint(3, 5), n)
        if len(s) != n:
            return None
        cells = place(rng, occ, H, W, s)
        if cells is None:
            return None
        a, b = rng.sample(range(1, 10), 2)
        cmap = {}
        for (r, c), (sr, sc) in zip(cells, s):
            v = a if rng.random() < 0.6 else b
            g[r][c] = v
            cmap[(sr, sc)] = v
        if i == 0:
            sh, sw = dims(s)
            crop = grid(sh, sw)
            for (sr, sc), v in cmap.items():
                crop[sr][sc] = v
    lr = [row[::-1] for row in crop]
    ud = crop[::-1]
    if len({str(crop), str(lr), str(ud)}) < 3:
        return None
    return g, (lr if axis == 0 else ud)


def generate(rng):
    axis = rng.randint(0, 1)
    pairs = []
    n = rng.randint(3, 4) + 1
    while len(pairs) < n:
        p = make_pair(rng, axis)
        if p:
            pairs.append({"input": p[0], "output": p[1]})
    return {"train": pairs[:-1], "test": pairs[-1:]}
