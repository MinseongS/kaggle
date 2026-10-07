import random

CONCEPT = ("one border side is a solid wall of colour K; every cell falls toward that wall and stacks against it, "
           "then the outermost cell of each stack is recoloured to K")


def rot(g):  # clockwise
    return [list(r) for r in zip(*g[::-1])]


def generate(rng: random.Random) -> dict:
    K = rng.randint(1, 9)
    others = [c for c in range(1, 10) if c != K]

    def make(side):
        while True:
            h, w = rng.randint(10, 16), rng.randint(10, 16)
            if side % 2:
                h, w = w, h  # keep final sizes varied after rotation
            g = [[0] * w for _ in range(h)]
            g[h - 1] = [K] * w
            pal = rng.sample(others, rng.randint(2, 3))
            dens = rng.uniform(0.12, 0.3)
            for r in range(h - 1):
                for c in range(w):
                    if rng.random() < dens:
                        g[r][c] = rng.choice(pal)
            out = [[0] * w for _ in range(h)]
            out[h - 1] = [K] * w
            for c in range(w):
                stack = [g[r][c] for r in range(h - 2, -1, -1) if g[r][c]]
                for i, v in enumerate(stack):
                    out[h - 2 - i][c] = v
                if stack:
                    out[h - 1 - len(stack)][c] = K
            if out == g or sum(v != 0 for r in g[:-1] for v in r) < 6:
                continue
            for _ in range(side):
                g, out = rot(g), rot(out)
            return g, out

    n = rng.randint(3, 4)
    while True:
        sides = [rng.randrange(4) for _ in range(n + 1)]
        if len(set(sides[:-1])) >= 2:
            break
    while True:
        pairs = [dict(zip(("input", "output"), make(s))) for s in sides]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
