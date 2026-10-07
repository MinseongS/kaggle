import random

CONCEPT = ("a full-length line of a task-specific colour splits the grid; every object on one side is mirrored "
           "across the line, the copy painted in a task-specific way (same colour, or a fixed recolour)")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def _shape(rng, n):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        if abs(r + dr) <= 1 and abs(c + dc) <= 1:
            cells.add((r + dr, c + dc))
    mr, mc = min(r for r, _ in cells), min(c for _, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def generate(rng: random.Random) -> dict:
    line = rng.randint(1, 9)
    copy_col = rng.choice([None, rng.choice([c for c in range(1, 10) if c != line])])
    pal = [c for c in range(1, 10) if c not in (line, copy_col)]

    def make():
        while True:
            # build in "horizontal line, objects above" frame, then transform
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            k = rng.randint(4, h - 5)
            side = min(k, h - 1 - k)  # rows available on each side for mirroring
            g = [[0] * w for _ in range(h)]
            for c in range(w):
                g[k][c] = line
            out = [row[:] for row in g]
            occ = set()
            n_obj, placed, tries = rng.randint(2, 4), 0, 0
            while placed < n_obj and tries < 200:
                tries += 1
                sh = _shape(rng, rng.randint(3, 6))
                hh = max(r for r, _ in sh) + 1
                ww = max(c for _, c in sh) + 1
                if hh > side:
                    continue
                r0 = rng.randint(k - side, k - hh)
                c0 = rng.randint(0, w - ww)
                cells = [(r0 + r, c0 + c) for r, c in sh]
                halo = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if halo & occ:
                    continue
                occ |= set(cells)
                col = rng.choice(pal)
                for r, c in cells:
                    g[r][c] = col
                    out[r][c] = col
                    out[2 * k - r][c] = copy_col if copy_col else col
                placed += 1
            if placed < 2:
                continue
            t = rng.randint(0, 3)  # 0 above, 1 below, 2 left, 3 right
            def tf(m):
                if t == 1:
                    m = m[::-1]
                elif t == 2:
                    m = [list(x) for x in zip(*m)]
                elif t == 3:
                    m = [list(x)[::-1] for x in zip(*m)]
                return [list(x) for x in m]
            return tf(g), tf(out)

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
