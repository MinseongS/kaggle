import random

CONCEPT = ("crop the area enclosed by the frame of colour F, then mirror it left-right if the marker cell sits "
           "left/right of the frame, or top-bottom if the marker sits above/below it")


def generate(rng: random.Random) -> dict:
    F, M = rng.sample(range(1, 10), 2)
    others = [c for c in range(1, 10) if c not in (F, M)]

    def make(axis):
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            ih, iw = rng.randint(3, 6), rng.randint(3, 6)
            fr, fc = rng.randint(1, h - ih - 3), rng.randint(1, w - iw - 3)
            g = [[0] * w for _ in range(h)]
            pal = rng.sample(others, rng.randint(2, 3))
            dens = rng.uniform(0.45, 0.8)
            inner = [[rng.choice(pal) if rng.random() < dens else 0 for _ in range(iw)] for _ in range(ih)]
            flipped = [row[::-1] for row in inner] if axis == "h" else inner[::-1]
            if flipped == inner or sum(v != 0 for r in inner for v in r) < 3:
                continue
            # noise outside the frame
            for r in range(h):
                for c in range(w):
                    if fr - 1 <= r <= fr + ih + 2 and fc - 1 <= c <= fc + iw + 2:
                        continue
                    if rng.random() < 0.12:
                        g[r][c] = rng.choice(others)
            for r in range(ih + 2):
                for c in range(iw + 2):
                    if r in (0, ih + 1) or c in (0, iw + 1):
                        g[fr + r][fc + c] = F
                    else:
                        g[fr + r][fc + c] = inner[r - 1][c - 1]
            if axis == "h":
                mr = fr + rng.randint(1, ih)
                mc = fc - 1 if rng.random() < 0.5 else fc + iw + 2
            else:
                mc = fc + rng.randint(1, iw)
                mr = fr - 1 if rng.random() < 0.5 else fr + ih + 2
            g[mr][mc] = M
            return g, flipped

    n = rng.randint(3, 4)
    axes = ["h", "v"] + [rng.choice("hv") for _ in range(n - 2)]
    rng.shuffle(axes)
    axes.append(rng.choice("hv"))
    while True:
        pairs = [dict(zip(("input", "output"), make(a))) for a in axes]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
