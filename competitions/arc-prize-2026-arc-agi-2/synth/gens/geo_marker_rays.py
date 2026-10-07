import random

CONCEPT = ("each coloured source cell has one marker cell (task colour) beside it; a ray of the source colour shoots "
           "from the source away from its marker until it reaches the border or any non-background cell (walls stop rays)")

D4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    marker, wall = rng.sample(range(1, 10), 2)
    pal = [c for c in range(1, 10) if c not in (marker, wall)]
    use_walls = rng.random() < 0.8

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            g = [[0] * w for _ in range(h)]
            if use_walls:
                for _ in range(rng.randint(1, 4)):
                    if rng.random() < 0.5:
                        L = rng.randint(2, max(2, w // 2))
                        r, c = rng.randrange(h), rng.randint(0, w - L)
                        for cc in range(c, c + L):
                            g[r][cc] = wall
                    else:
                        L = rng.randint(2, max(2, h // 2))
                        r, c = rng.randint(0, h - L), rng.randrange(w)
                        for rr in range(r, r + L):
                            g[rr][c] = wall
            srcs = []
            for _ in range(rng.randint(2, 4)):
                r, c = rng.randrange(h), rng.randrange(w)
                dr, dc = rng.choice(D4)
                mr, mc = r - dr, c - dc
                if not (0 <= mr < h and 0 <= mc < w) or g[r][c] or g[mr][mc]:
                    continue
                g[r][c] = rng.choice(pal)
                g[mr][mc] = marker
                srcs.append((r, c, dr, dc, mr, mc))
            if len(srcs) < 2:
                continue
            # each source has exactly one marker neighbour and vice versa
            bad = False
            src_set = {(s[0], s[1]) for s in srcs}
            for r, c, dr, dc, mr, mc in srcs:
                if sum(1 for a, b in D4 if 0 <= r + a < h and 0 <= c + b < w and g[r + a][c + b] == marker) != 1:
                    bad = True
                if sum(1 for a, b in D4 if (mr + a, mc + b) in src_set) != 1:
                    bad = True
                # sources must not touch walls/other sources (keeps them unambiguous)
                if any(0 <= r + a < h and 0 <= c + b < w and g[r + a][c + b] not in (0, marker)
                       for a, b in D4):
                    bad = True
            if bad:
                continue
            out = [row[:] for row in g]
            used = set()
            for r, c, dr, dc, _, _ in srcs:
                path = []
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w and g[rr][cc] == 0:
                    path.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(path) < 2 or used & set(path):
                    bad = True
                    break
                used |= set(path)
                for a, b in path:
                    out[a][b] = g[r][c]
            if not bad:
                return g, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
