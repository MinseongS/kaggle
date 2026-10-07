import random

CONCEPT = "a legend strip of colour pairs (a->b) behind a divider line recolours the objects in the main area; colours absent from the legend stay"


def _shape(rng, bh, bw, n):
    cells = {(rng.randrange(bh), rng.randrange(bw))}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        if 0 <= r + dr < bh and 0 <= c + dc < bw:
            cells.add((r + dr, c + dc))
    return sorted(cells)


def generate(rng: random.Random) -> dict:
    D = rng.randint(1, 9)
    top = rng.random() < 0.5
    palette = [c for c in range(1, 10) if c != D]

    def make():
        while True:
            k = rng.randint(2, 3)
            cols = rng.sample(palette, 2 * k + 1)
            src, dst, distr = cols[:k], cols[k:2 * k], cols[2 * k]
            h, w = rng.randint(11, 18), rng.randint(13, 19)
            if 2 * k + 1 > h:
                continue
            g = [[0] * w for _ in range(h)]
            r0 = rng.randint(0, h - 2 * k)
            for i in range(k):
                g[r0 + 2 * i][0] = src[i]
                g[r0 + 2 * i][1] = dst[i]
            for r in range(h):
                g[r][2] = D
            occ = set()
            objs = []
            want = [*src, *rng.sample(src, rng.randint(0, k)), *([distr] * rng.randint(0, 2))]
            ok = True
            for col in want:
                placed = False
                for _ in range(60):
                    bh, bw = rng.randint(2, 3), rng.randint(2, 3)
                    sh = _shape(rng, bh, bw, rng.randint(3, bh * bw))
                    r, c = rng.randint(0, h - bh), rng.randint(4, w - bw)
                    cells = [(r + a, c + b) for a, b in sh]
                    if any((y + dy, x + dx) in occ for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                        continue
                    occ.update(cells)
                    objs.append((cells, col))
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue
            out = [row[:] for row in g]
            m = dict(zip(src, dst))
            for cells, col in objs:
                for y, x in cells:
                    g[y][x] = col
                    out[y][x] = m.get(col, col)
            if top:
                g = [list(r) for r in zip(*g)]
                out = [list(r) for r in zip(*out)]
            return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
