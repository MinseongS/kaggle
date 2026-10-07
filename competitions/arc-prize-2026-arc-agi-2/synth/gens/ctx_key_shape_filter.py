import random

CONCEPT = "a framed key in the corner shows a shape and colour; only objects with exactly that shape survive and take the key colour, all others vanish"


def _shape(rng, n):
    cells = {(1, 1)}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        if 0 <= r + dr < 3 and 0 <= c + dc < 3:
            cells.add((r + dr, c + dc))
    return _norm(cells)


def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def _canon(sh):
    forms = []
    cur = list(sh)
    for _ in range(4):
        cur = [(c, -r) for r, c in cur]
        forms.append(_norm(cur))
        forms.append(_norm([(r, -c) for r, c in cur]))
    return min(forms)


def generate(rng: random.Random) -> dict:
    F = rng.randint(1, 9)
    palette = [c for c in range(1, 10) if c != F]

    def make():
        while True:
            h, w = rng.randint(12, 18), rng.randint(12, 18)
            key = _shape(rng, rng.randint(3, 5))
            kc = rng.choice(palette)
            others = []
            for _ in range(50):
                s = _shape(rng, rng.randint(3, 5))
                if _canon(s) != _canon(key) and s not in others:
                    others.append(s)
                if len(others) >= 3:
                    break
            if len(others) < 2:
                continue
            g = [[0] * w for _ in range(h)]
            for r, c in key:
                g[r][c] = kc
            for i in range(4):
                g[3][i] = F
                g[i][3] = F
            occ = set((a, b) for a in range(5) for b in range(5))
            want = [key] * rng.randint(1, 3) + [rng.choice(others) for _ in range(rng.randint(3, 5))]
            objs = []
            for sh in want:
                for _ in range(80):
                    r, c = rng.randint(0, h - 3), rng.randint(0, w - 3)
                    cells = [(r + a, c + b) for a, b in sh]
                    if any(y >= h or x >= w for y, x in cells):
                        continue
                    if any((y + dy, x + dx) in occ for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                        continue
                    occ.update(cells)
                    objs.append((cells, sh == key, rng.choice(palette)))
                    break
            if sum(o[1] for o in objs) < 1 or sum(not o[1] for o in objs) < 2:
                continue
            out = [row[:] for row in g]
            for cells, keep, col in objs:
                for y, x in cells:
                    g[y][x] = col
                    out[y][x] = kc if keep else 0
            return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
