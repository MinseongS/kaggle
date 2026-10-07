import random

CONCEPT = "a key cell in the top-left corner gives a colour; only objects of that colour fall down until they hit the floor or another cell"


def generate(rng: random.Random) -> dict:
    def make(h, w, key, colors):
        while True:
            g = [[0] * w for _ in range(h)]
            g[0][0] = key
            cells = [(r, c) for r in range(2, h - 1) for c in range(1, w)]
            rng.shuffle(cells)
            for r, c in cells[: rng.randint(h * w // 10, h * w // 5)]:
                g[r][c] = rng.choice(colors)
            out = [row[:] for row in g]
            for c in range(1, w):
                for r in range(h - 2, -1, -1):  # bottom-up so stacks settle
                    if out[r][c] == key and r > 0:
                        rr = r
                        while rr + 1 < h and out[rr + 1][c] == 0:
                            rr += 1
                        out[r][c], out[rr][c] = 0, key
            if out != g:
                return g, out

    colors = rng.sample(range(1, 10), 3)
    key = colors[0]
    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        h, w = rng.randint(8, 14), rng.randint(8, 14)
        i, o = make(h, w, key, colors)
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
