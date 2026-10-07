import random

CONCEPT = "several panels separated by divider lines are identical except one; output that odd panel"


def generate(rng: random.Random) -> dict:
    D = rng.randint(1, 9)
    palette = [c for c in range(1, 10) if c != D]

    def make():
        while True:
            s = rng.randint(3, 6)
            pr, pc = rng.choice([(1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (3, 3), (3, 1), (4, 1), (2, 4)])
            if pr * pc < 3:
                continue
            H, W = pr * s + pr - 1, pc * s + pc - 1
            if H > 30 or W > 30:
                continue
            cols = rng.sample(palette, rng.randint(2, 3))
            base = [[rng.choice(cols) if rng.random() < 0.55 else 0 for _ in range(s)] for _ in range(s)]
            odd = [row[:] for row in base]
            for _ in range(rng.randint(1, 2)):
                a, b = rng.randrange(s), rng.randrange(s)
                odd[a][b] = rng.choice([v for v in [0] + cols if v != base[a][b]])
            if odd == base:
                continue
            g = [[D] * W for _ in range(H)]
            k = rng.randrange(pr * pc)
            for i in range(pr):
                for j in range(pc):
                    p = odd if i * pc + j == k else base
                    for a in range(s):
                        for b in range(s):
                            g[i * (s + 1) + a][j * (s + 1) + b] = p[a][b]
            return g, odd

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
