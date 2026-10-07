import random

CONCEPT = ("hollow rectangles whose interior is larger than a hidden size N get filled with their own border colour; "
           "rectangles with smaller interiors are erased")


def generate(rng: random.Random) -> dict:
    N = rng.randint(3, 14)
    bg = 0 if rng.random() < 0.8 else rng.randint(1, 9)
    pal = [c for c in range(10) if c != bg]

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            g = [[bg] * w for _ in range(h)]
            out = [[bg] * w for _ in range(h)]
            blocked = set()
            areas = []
            for _ in range(rng.randint(3, 6)):
                for _try in range(40):
                    rh, rw = rng.randint(3, 8), rng.randint(3, 8)
                    r0, c0 = rng.randint(0, h - rh), rng.randint(0, w - rw)
                    cells = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
                    if any(x in blocked for x in cells):
                        continue
                    for r in range(r0 - 1, r0 + rh + 1):
                        for c in range(c0 - 1, c0 + rw + 1):
                            blocked.add((r, c))
                    col = rng.choice(pal)
                    a = (rh - 2) * (rw - 2)
                    areas.append(a)
                    for r, c in cells:
                        edge = r in (r0, r0 + rh - 1) or c in (c0, c0 + rw - 1)
                        if edge:
                            g[r][c] = col
                        if a > N:
                            out[r][c] = col
                    break
            if len(areas) >= 3 and any(a > N for a in areas) and any(a <= N for a in areas):
                return g, out, areas

    n = rng.randint(3, 4)
    train, alls = [], []
    for _ in range(n):
        g, o, areas = make()
        train.append({"input": g, "output": o})
        alls += areas
    lo = max(a for a in alls if a <= N)
    hi = min(a for a in alls if a > N)
    while True:
        g, o, areas = make()
        if all(a <= lo or a >= hi for a in areas) and all(o != p["output"] for p in train):
            return {"train": train, "test": [{"input": g, "output": o}]}
