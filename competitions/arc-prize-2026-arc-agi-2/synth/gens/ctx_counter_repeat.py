import random

CONCEPT = "the number of counter dots in the top row says how many copies of each object to lay out to the right (task-fixed gap between copies)"


def generate(rng: random.Random) -> dict:
    K = rng.randint(1, 9)
    gap = rng.randint(1, 2)
    palette = [c for c in range(1, 10) if c != K]
    counts = []

    def make(n):
        while True:
            nobj = rng.randint(1, 3)
            objs = []
            for _ in range(nobj):
                bh, bw = rng.randint(2, 3), rng.randint(1, 3)
                cols = rng.sample(palette, rng.randint(1, 2))
                sh = [[rng.choice(cols) if rng.random() < 0.75 else 0 for _ in range(bw)] for _ in range(bh)]
                if not any(sh[0]) or not any(sh[-1]) or not any(r[0] for r in sh) or not any(r[-1] for r in sh):
                    break
                objs.append(sh)
            if len(objs) != nobj:
                continue
            maxw = max(len(o[0]) for o in objs)
            need = 1 + n * maxw + (n - 1) * gap
            w = rng.randint(max(need, 2 * n), min(30, need + 4))
            if need > 30:
                continue
            T = sum(len(o) + 1 for o in objs) - 1
            h = max(8, 2 + T + rng.randint(0, 4))
            if h > 30:
                continue
            g = [[0] * w for _ in range(h)]
            for i in range(n):
                g[0][2 * i] = K
            out = [row[:] for row in g]
            r = 2 + rng.randint(0, h - 2 - T)
            for o in objs:
                c0 = rng.randint(0, 1)
                bh, bw = len(o), len(o[0])
                for k in range(n):
                    for a in range(bh):
                        for b in range(bw):
                            if o[a][b]:
                                if k == 0:
                                    g[r + a][c0 + b] = o[a][b]
                                out[r + a][c0 + k * (bw + gap) + b] = o[a][b]
                r += bh + 1
            return g, out

    npairs = rng.randint(3, 4) + 1
    while True:
        counts = [rng.randint(2, 5) for _ in range(npairs)]
        if len(set(counts[:-1])) >= 2:
            break
    pairs = []
    for n in counts:
        i, o = make(n)
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
