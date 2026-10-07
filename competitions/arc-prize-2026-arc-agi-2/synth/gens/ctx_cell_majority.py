import random

CONCEPT = "divider lines split the grid into cells; each cell becomes its majority colour (task-level choice: fill in place or shrink to one pixel per cell)"


def generate(rng: random.Random) -> dict:
    D = rng.randint(1, 9)
    shrink = rng.random() < 0.5
    palette = [c for c in range(1, 10) if c != D]

    def make():
        while True:
            s = rng.randint(3, 5)
            k, m = rng.randint(2, 4), rng.randint(2, 4)
            h, w = k * s + k - 1, m * s + m - 1
            if h > 30 or w > 30:
                continue
            g = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if r % (s + 1) == s or c % (s + 1) == s:
                        g[r][c] = D
            small = [[0] * m for _ in range(k)]
            mixed = 0
            for i in range(k):
                for j in range(m):
                    if rng.random() < 0.1:
                        continue
                    nc = rng.choice([1, 2, 2, 3])
                    cols = rng.sample(palette, nc)
                    top = rng.randint(2, max(2, s * s // 2))
                    counts = [top] + [rng.randint(1, top - 1) for _ in cols[1:]]
                    if sum(counts) > s * s:
                        counts = [top] + [1] * (nc - 1)
                    if nc > 1:
                        mixed += 1
                    cells = [(a, b) for a in range(s) for b in range(s)]
                    rng.shuffle(cells)
                    idx = 0
                    for col, cnt in zip(cols, counts):
                        for a, b in cells[idx:idx + cnt]:
                            g[i * (s + 1) + a][j * (s + 1) + b] = col
                        idx += cnt
                    small[i][j] = cols[0]
            if mixed < 2:
                continue
            if shrink:
                out = small
            else:
                out = [row[:] for row in g]
                for i in range(k):
                    for j in range(m):
                        for a in range(s):
                            for b in range(s):
                                out[i * (s + 1) + a][j * (s + 1) + b] = small[i][j]
            if out != g:
                return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
