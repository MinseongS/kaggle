import random

CONCEPT = ("through each object's centre draw a full horizontal and vertical line in the object's colour over the "
           "background; where lines of two different objects cross, paint a fixed colour X")

SHAPES = [
    ["1"], ["111"], ["1", "1", "1"], ["010", "111", "010"], ["101", "010", "101"], ["111", "111", "111"],
    ["111", "101", "111"], ["11111"], ["00100", "01110", "11111"],
]


def generate(rng: random.Random) -> dict:
    X = rng.randint(1, 9)
    pal = [c for c in range(1, 10) if c != X]

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            g = [[0] * w for _ in range(h)]
            objs, blocked = [], set()
            for col in rng.sample(pal, rng.randint(2, 4)):
                shp = rng.choice(SHAPES)
                if rng.random() < 0.5:
                    shp = ["".join(r[::-1]) for r in shp][::-1]
                sh, sw = len(shp), len(shp[0])
                for _try in range(30):
                    r0, c0 = rng.randint(1, h - sh - 1), rng.randint(1, w - sw - 1)
                    cells = [(r0 + r, c0 + c) for r in range(sh) for c in range(sw) if shp[r][c] == "1"]
                    rows, cs = set(range(r0, r0 + sh)), set(range(c0, c0 + sw))
                    if any(rows & o["rows"] or cs & o["cols"] for o in objs):
                        continue
                    objs.append({"cells": cells, "col": col, "rows": {x for r in rows for x in (r - 1, r, r + 1)},
                                 "cols": {x for c in cs for x in (c - 1, c, c + 1)},
                                 "cr": r0 + sh // 2, "cc": c0 + sw // 2})
                    for r, c in cells:
                        g[r][c] = col
                    break
            if len(objs) < 2:
                continue
            out = [row[:] for row in g]
            for o in objs:
                for c in range(w):
                    if g[o["cr"]][c] == 0:
                        out[o["cr"]][c] = o["col"]
                for r in range(h):
                    if g[r][o["cc"]] == 0:
                        out[r][o["cc"]] = o["col"]
            for a in objs:
                for b in objs:
                    if a is not b:
                        out[a["cr"]][b["cc"]] = X
            return g, out

    n = rng.randint(3, 4)
    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(n + 1)]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
