import random

CONCEPT = ("the grid holds one multi-coloured object and N scattered marker cells of a task colour; output the "
           "object cropped to its bounding box and scaled up by factor N")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    marker = rng.randint(1, 9)
    pal = [c for c in range(1, 10) if c != marker]

    def make(n):
        while True:
            oh, ow = rng.randint(2, 4), rng.randint(2, 4)
            if oh * n > 30 or ow * n > 30:
                continue
            cols = rng.sample(pal, rng.randint(2, 3))
            obj = [[rng.choice(cols) if rng.random() < 0.7 else 0 for _ in range(ow)] for _ in range(oh)]
            # tight bbox and at least 2 colours
            if not any(obj[0]) or not any(obj[-1]) or not any(r[0] for r in obj) or not any(r[-1] for r in obj):
                continue
            if len({v for r in obj for v in r if v}) < 2:
                continue
            # object must be 8-connected
            cells = {(r, c) for r in range(oh) for c in range(ow) if obj[r][c]}
            seen, st = set(), [next(iter(cells))]
            while st:
                r, c = st.pop()
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                st += [(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (r + a, c + b) in cells]
            if seen != cells:
                continue
            h, w = rng.randint(10, 16), rng.randint(10, 16)
            g = [[0] * w for _ in range(h)]
            r0, c0 = rng.randint(0, h - oh), rng.randint(0, w - ow)
            for r in range(oh):
                for c in range(ow):
                    g[r0 + r][c0 + c] = obj[r][c]
            near = {(r, c) for r in range(r0 - 1, r0 + oh + 1) for c in range(c0 - 1, c0 + ow + 1)}
            free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in near]
            rng.shuffle(free)
            for r, c in free[:n]:
                g[r][c] = marker
            out = [[obj[r // n][c // n] for c in range(ow * n)] for r in range(oh * n)]
            return g, out

    while True:
        k = rng.randint(3, 4) + 1
        ns = [rng.randint(2, 4) for _ in range(k)]
        if len(set(ns[:-1])) < 2:
            continue
        pairs = [dict(zip(("input", "output"), make(n))) for n in ns]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
