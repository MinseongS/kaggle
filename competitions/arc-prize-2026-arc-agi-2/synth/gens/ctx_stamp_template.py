import random

CONCEPT = "a framed template in the corner is stamped centred on every marker cell, in the marker's colour"


def generate(rng: random.Random) -> dict:
    p = rng.choice([3, 3, 5])
    F, T = rng.sample(range(1, 10), 2)
    palette = [c for c in range(1, 10) if c not in (F, T)]

    def template():
        while True:
            t = [[1 if rng.random() < 0.55 else 0 for _ in range(p)] for _ in range(p)]
            t[p // 2][p // 2] = 1
            n = sum(map(sum, t))
            if p * 1 + 1 <= n < p * p and any(t[0]) and any(t[-1]) and any(r[0] for r in t) and any(r[-1] for r in t):
                return t

    def make():
        while True:
            h, w = rng.randint(12, 19), rng.randint(12, 19)
            t = template()
            g = [[0] * w for _ in range(h)]
            for a in range(p):
                for b in range(p):
                    if t[a][b]:
                        g[a][b] = T
            for i in range(p + 1):
                g[p][i] = F
                g[i][p] = F
            occ = set((a, b) for a in range(p + 1) for b in range(p + 1))
            n = rng.randint(2, 4)
            marks = []
            for _ in range(200):
                if len(marks) == n:
                    break
                r, c = rng.randint(p // 2, h - 1 - p // 2), rng.randint(p // 2, w - 1 - p // 2)
                hp = p // 2
                box = [(r + a, c + b) for a in range(-hp - 1, hp + 2) for b in range(-hp - 1, hp + 2)]
                if any(x in occ for x in box):
                    continue
                occ.update(box)
                marks.append((r, c, rng.choice(palette)))
            if len(marks) < 2:
                continue
            out = [row[:] for row in g]
            for r, c, col in marks:
                g[r][c] = col
                for a in range(p):
                    for b in range(p):
                        if t[a][b]:
                            out[r + a - p // 2][c + b - p // 2] = col
            return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
