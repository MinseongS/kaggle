import random

CONCEPT = "the border frame colour names the noise colour: noise pixels of that colour are removed (restored to the rectangle or background beneath), other noise stays"


def generate(rng: random.Random) -> dict:
    P = rng.sample(range(1, 10), 3)
    palette = [c for c in range(1, 10) if c not in P]

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            F = rng.choice(P)
            base = [[0] * w for _ in range(h)]
            occ = set()
            nrect = 0
            for _ in range(60):
                if nrect >= rng.randint(2, 4):
                    break
                rh, rw = rng.randint(3, 6), rng.randint(3, 6)
                r, c = rng.randint(2, h - 2 - rh), rng.randint(2, w - 2 - rw)
                if r < 2 or c < 2:
                    continue
                cells = [(r + a, c + b) for a in range(rh) for b in range(rw)]
                if any((y + dy, x + dx) in occ for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                    continue
                occ.update(cells)
                col = rng.choice(palette)
                for y, x in cells:
                    base[y][x] = col
                nrect += 1
            if nrect < 2:
                continue
            for i in range(h):
                for j in range(w):
                    if i in (0, h - 1) or j in (0, w - 1):
                        base[i][j] = F
            g = [row[:] for row in base]
            dens = rng.uniform(0.06, 0.12)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if rng.random() < dens:
                        g[i][j] = rng.choice(P)
            out = [row[:] for row in g]
            nF = nO = nFin = 0
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if g[i][j] == F:
                        out[i][j] = base[i][j]
                        nF += 1
                        nFin += base[i][j] != 0
                    elif g[i][j] in P:
                        nO += 1
            if nF >= 3 and nO >= 3 and nFin >= 1:
                return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
