import random

CONCEPT = "one panel shows a full multicolour motif; the other panels show only its anchor-colour cell, and the motif is completed around each anchor"


def generate(rng: random.Random) -> dict:
    D, A = rng.sample(range(1, 10), 2)
    vertical = rng.random() < 0.5
    palette = [c for c in range(1, 10) if c not in (D, A)]

    def make():
        while True:
            n = rng.randint(3, 4)
            S = rng.randint(5, 7)
            ph, pw = rng.randint(2, 3), rng.randint(2, 3)
            if ph * pw < 4:
                continue
            cols = rng.sample(palette, rng.randint(1, 3))
            pat = [[rng.choice(cols) if rng.random() < 0.8 else 0 for _ in range(pw)] for _ in range(ph)]
            ar, ac = rng.randrange(ph), rng.randrange(pw)
            pat[ar][ac] = A
            if sum(1 for row in pat for v in row if v) < 4:
                continue
            if not any(pat[0]) or not any(pat[-1]) or not any(r[0] for r in pat) or not any(r[-1] for r in pat):
                continue
            H, W = S, n * S + n - 1
            if W > 30:
                continue
            g = [[0] * W for _ in range(H)]
            for j in range(1, n):
                for r in range(H):
                    g[r][j * (S + 1) - 1] = D
            out = [row[:] for row in g]
            full = rng.randrange(n)
            seen = set()
            for j in range(n):
                while True:
                    r, c = rng.randint(0, S - ph), rng.randint(0, S - pw)
                    if (r, c) not in seen or len(seen) > 5:
                        break
                seen.add((r, c))
                x0 = j * (S + 1)
                for a in range(ph):
                    for b in range(pw):
                        if pat[a][b]:
                            out[r + a][x0 + c + b] = pat[a][b]
                            if j == full or (a, b) == (ar, ac):
                                g[r + a][x0 + c + b] = pat[a][b]
            if vertical:
                g = [list(x) for x in zip(*g)]
                out = [list(x) for x in zip(*out)]
            return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
