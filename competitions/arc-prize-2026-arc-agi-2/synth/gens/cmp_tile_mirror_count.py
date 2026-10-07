import random

CONCEPT = ("tile the input k x k times where k is the number of distinct non-background colours; alternate tiles "
           "are mirrored (mirror pattern fixed per task)")


def generate(rng: random.Random) -> dict:
    mode = rng.choice(["hv", "rot", "lr", "ud"])

    def tile(g, i, j):
        lr = lambda x: [r[::-1] for r in x]
        ud = lambda x: x[::-1]
        if mode == "hv":
            if j % 2: g = lr(g)
            if i % 2: g = ud(g)
        elif mode == "rot":
            if (i + j) % 2: g = ud(lr(g))
        elif mode == "lr":
            if (i + j) % 2: g = lr(g)
        else:
            if (i + j) % 2: g = ud(g)
        return g

    def make(k):
        while True:
            hs, ws = rng.randint(3, 5), rng.randint(3, 5)
            cols = rng.sample(range(1, 10), k)
            p0 = rng.uniform(0.15, 0.45)
            g = [[0 if rng.random() < p0 else rng.choice(cols) for _ in range(ws)] for _ in range(hs)]
            if len({v for r in g for v in r} - {0}) != k:
                continue
            if g == [r[::-1] for r in g] or g == g[::-1]:
                continue
            out = [[0] * (ws * k) for _ in range(hs * k)]
            for i in range(k):
                for j in range(k):
                    t = tile(g, i, j)
                    for r in range(hs):
                        for c in range(ws):
                            out[i * hs + r][j * ws + c] = t[r][c]
            return g, out

    n = rng.randint(3, 4)
    while True:
        ks = [rng.randint(2, 4) for _ in range(n + 1)]
        if len(set(ks[:-1])) >= 2:
            break
    while True:
        pairs = [dict(zip(("input", "output"), make(k))) for k in ks]
        ins = [str(p["input"]) for p in pairs]
        if len(set(ins)) == len(ins) and all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
