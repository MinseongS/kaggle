import random

CONCEPT = ("every colour that appears exactly twice marks opposite corners of a rectangle: draw its outline in that "
           "colour and fill the interior with a task-specific colour (or leave it empty); singleton cells are distractors")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    fill = rng.choice([0, rng.randint(1, 9)])
    pal = [c for c in range(1, 10) if c != fill]
    distract = rng.random() < 0.8

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            g = [[0] * w for _ in range(h)]
            out = [[0] * w for _ in range(h)]
            n = rng.randint(2, 4)
            cols = rng.sample(pal, min(len(pal), n + 3))
            occ = set()
            rects = 0
            for col in cols[:n]:
                for _ in range(50):
                    rh, rw = rng.randint(3, 7), rng.randint(3, 7)
                    r1, c1 = rng.randint(0, h - rh), rng.randint(0, w - rw)
                    r2, c2 = r1 + rh - 1, c1 + rw - 1
                    area = {(r, c) for r in range(r1 - 1, r2 + 2) for c in range(c1 - 1, c2 + 2)}
                    if area & occ:
                        continue
                    occ |= area
                    if rng.random() < 0.5:
                        g[r1][c1] = g[r2][c2] = col
                    else:
                        g[r1][c2] = g[r2][c1] = col
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            edge = r in (r1, r2) or c in (c1, c2)
                            out[r][c] = col if edge else fill
                    rects += 1
                    break
            if rects < 2:
                continue
            if distract:
                for col in cols[n:n + rng.randint(1, 3)]:
                    for _ in range(50):
                        r, c = rng.randrange(h), rng.randrange(w)
                        if (r, c) not in occ:
                            occ |= {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
                            g[r][c] = out[r][c] = col
                            break
            return g, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
