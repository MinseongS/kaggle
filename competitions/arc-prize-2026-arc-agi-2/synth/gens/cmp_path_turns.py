import random

CONCEPT = ("a start cell on the border sends a trail of its colour straight inward; on meeting a marker it turns "
           "left or right depending on the marker colour (marker stays), until it leaves the grid")

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # clockwise


def generate(rng: random.Random) -> dict:
    cols = rng.sample(range(1, 10), 4)
    bg = 0 if rng.random() < 0.75 else cols.pop()
    S, Lc, Rc = cols[:3]

    def simulate(g, start, d):
        h, w = len(g), len(g[0])
        out = [row[:] for row in g]
        pos, seen, used, steps, turns = start, set(), set(), 0, 0
        while True:
            if (pos, d) in seen:
                return None
            seen.add((pos, d))
            nr, nc = pos[0] + DIRS[d][0], pos[1] + DIRS[d][1]
            if not (0 <= nr < h and 0 <= nc < w):
                return out, used, steps, turns
            v = g[nr][nc]
            if v == Lc:
                d = (d - 1) % 4; used.add("L"); turns += 1
            elif v == Rc:
                d = (d + 1) % 4; used.add("R"); turns += 1
            else:
                pos = (nr, nc); out[nr][nc] = S; steps += 1

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            g = [[bg] * w for _ in range(h)]
            side = rng.randrange(4)
            if side == 0:
                start, d = (0, rng.randint(1, w - 2)), 2
            elif side == 1:
                start, d = (rng.randint(1, h - 2), w - 1), 3
            elif side == 2:
                start, d = (h - 1, rng.randint(1, w - 2)), 0
            else:
                start, d = (rng.randint(1, h - 2), 0), 1
            g[start[0]][start[1]] = S
            trail, cur, d0 = {start}, start, d
            for _ in range(rng.randint(2, 5)):
                for _s in range(rng.randint(2, 6)):
                    nxt = (cur[0] + DIRS[d][0], cur[1] + DIRS[d][1])
                    if not (0 <= nxt[0] < h and 0 <= nxt[1] < w) or nxt in trail:
                        break
                    cur = nxt; trail.add(cur)
                m = (cur[0] + DIRS[d][0], cur[1] + DIRS[d][1])
                if not (0 <= m[0] < h and 0 <= m[1] < w) or m in trail or g[m[0]][m[1]] != bg:
                    break
                t = rng.choice("LR")
                g[m[0]][m[1]] = Lc if t == "L" else Rc
                d = (d - 1) % 4 if t == "L" else (d + 1) % 4
            free = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if g[r][c] == bg and (r, c) not in trail]
            for r, c in rng.sample(free, rng.randint(1, 5)):
                g[r][c] = rng.choice((Lc, Rc))
            res = simulate(g, start, d0)
            if res is None:
                continue
            out, used, steps, turns = res
            if turns >= 2 and steps >= 8 and out != g:
                return g, out, used

    n = rng.randint(3, 4)
    while True:
        train, used = [], set()
        for _ in range(n):
            g, o, u = make()
            train.append({"input": g, "output": o})
            used |= u
        if used != {"L", "R"}:
            continue
        g, o, _ = make()
        if all(o != p["output"] for p in train):
            return {"train": train, "test": [{"input": g, "output": o}]}
