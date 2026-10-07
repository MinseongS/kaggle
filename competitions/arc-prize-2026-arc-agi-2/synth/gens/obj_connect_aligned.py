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


CONCEPT = "dots of several colours; two dots of the SAME colour in one row/column with clear space between are joined by a line (in their colour, or in a fixed per-task colour); differently coloured aligned dots stay unjoined"


def make_pair(rng, cols, line):
    H, W = rng.randint(10, 18), rng.randint(10, 18)
    dots = {}
    for _ in range(rng.randint(2, 4)):  # deliberate aligned same-colour pairs
        col = rng.choice(cols)
        if rng.random() < 0.5:
            r = rng.randrange(H)
            a, b = sorted(rng.sample(range(W), 2))
            ps = [(r, a), (r, b)]
        else:
            c = rng.randrange(W)
            a, b = sorted(rng.sample(range(H), 2))
            ps = [(a, c), (b, c)]
        for p in ps:
            dots[p] = col
    for _ in range(rng.randint(3, 7)):
        dots[(rng.randrange(H), rng.randrange(W))] = rng.choice(cols)
    pts = sorted(dots)
    for i, p in enumerate(pts):  # no touching dots
        for q in pts[i + 1:]:
            if abs(p[0] - q[0]) <= 1 and abs(p[1] - q[1]) <= 1:
                return None
    g = grid(H, W)
    for (r, c), v in dots.items():
        g[r][c] = v
    o = [row[:] for row in g]
    used = set()
    nlines = 0
    diff_aligned = 0
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            if p[0] == q[0]:
                seg = [(p[0], c) for c in range(p[1] + 1, q[1])]
            elif p[1] == q[1]:
                seg = [(r, p[1]) for r in range(p[0] + 1, q[0])]
            else:
                continue
            if any(s in dots for s in seg):
                continue
            if dots[p] != dots[q]:
                diff_aligned += 1
                continue
            if any(s in used for s in seg):
                return None
            used.update(seg)
            nlines += 1
            for r, c in seg:
                o[r][c] = dots[p] if line is None else line
    if nlines < 2:
        return None
    return g, o, diff_aligned


def generate(rng):
    allc = rng.sample(range(1, 10), 4)
    cols = allc[:3]
    line = None if rng.random() < 0.5 else allc[3]
    while True:
        pairs = []
        da = 0
        n = rng.randint(3, 4) + 1
        while len(pairs) < n:
            p = make_pair(rng, cols, line)
            if p:
                pairs.append({"input": p[0], "output": p[1]})
                if len(pairs) < n:
                    da += p[2]
        if da > 0:
            return {"train": pairs[:-1], "test": pairs[-1:]}
