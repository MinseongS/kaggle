import random

CONCEPT = "two divider lines cut the grid into four regions; every object takes the colour shown in the grid corner of the region it sits in"


def _shape(rng, bh, bw, n):
    cells = {(rng.randrange(bh), rng.randrange(bw))}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        if 0 <= r + dr < bh and 0 <= c + dc < bw:
            cells.add((r + dr, c + dc))
    return sorted(cells)


def generate(rng: random.Random) -> dict:
    D, C = rng.sample(range(1, 10), 2)
    palette = [c for c in range(1, 10) if c not in (D, C)]

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            r0, c0 = rng.randint(5, h - 6), rng.randint(5, w - 6)
            g = [[0] * w for _ in range(h)]
            for r in range(h):
                g[r][c0] = D
            for c in range(w):
                g[r0][c] = D
            corner = rng.sample(palette, 4)
            g[0][0], g[0][w - 1], g[h - 1][0], g[h - 1][w - 1] = corner
            regions = [(1, r0 - 1, 1, c0 - 1), (1, r0 - 1, c0 + 1, w - 2),
                       (r0 + 1, h - 2, 1, c0 - 1), (r0 + 1, h - 2, c0 + 1, w - 2)]
            out = [row[:] for row in g]
            occ = set()
            filled = 0
            for qi, (ra, rb, ca, cb) in enumerate(regions):
                got = 0
                for _ in range(rng.randint(1, 3) * 20):
                    if got >= 3:
                        break
                    bh, bw = rng.randint(1, 3), rng.randint(1, 3)
                    if rb - ra + 1 < bh or cb - ca + 1 < bw:
                        continue
                    sh = _shape(rng, bh, bw, rng.randint(max(1, bh * bw // 2), bh * bw))
                    r, c = rng.randint(ra, rb - bh + 1), rng.randint(ca, cb - bw + 1)
                    cells = [(r + a, c + b) for a, b in sh]
                    if any((y + dy, x + dx) in occ for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                        continue
                    if any(y in (r0 - 1, r0 + 1) or x in (c0 - 1, c0 + 1) for y, x in cells) and rng.random() < 0.7:
                        continue
                    occ.update(cells)
                    got += 1
                    for y, x in cells:
                        g[y][x] = C
                        out[y][x] = corner[qi]
                filled += got > 0
            if filled >= 3:
                return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
