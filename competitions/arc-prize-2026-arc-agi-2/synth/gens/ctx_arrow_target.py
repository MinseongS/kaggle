import random

CONCEPT = "each block has a one-cell tip pointing at a target square; a line is drawn from the tip to the target and the target takes the block's colour; unpointed targets stay"

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def generate(rng: random.Random) -> dict:
    T = rng.randint(1, 9)
    palette = [c for c in range(1, 10) if c != T]

    def free(occ, cells, h, w):
        return all(0 <= y < h and 0 <= x < w and (y, x) not in occ for y, x in cells)

    def halo(cells):
        return {(y + a, x + b) for y, x in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}

    def make():
        while True:
            h, w = rng.randint(13, 20), rng.randint(13, 20)
            g = [[0] * w for _ in range(h)]
            out = [[0] * w for _ in range(h)]
            occ = set()
            cols = rng.sample(palette, 3)
            npt = rng.randint(1, 3)
            done = 0
            for col in cols[:npt]:
                for _ in range(100):
                    dr, dc = rng.choice(DIRS)
                    r, c = rng.randint(0, h - 3), rng.randint(0, w - 3)
                    block = [(r + a, c + b) for a in range(3) for b in range(3)]
                    tip = (r + 1 + 2 * dr, c + 1 + 2 * dc)
                    d = rng.randint(2, 6)
                    ray = [(tip[0] + dr * k, tip[1] + dc * k) for k in range(1, d + 1)]
                    end = (tip[0] + dr * (d + 1), tip[1] + dc * (d + 1))
                    # 2x2 target containing `end`, extending along the direction
                    if dr:
                        ty = end[0] if dr > 0 else end[0] - 1
                        tx = end[1] - rng.randint(0, 1)
                    else:
                        tx = end[1] if dc > 0 else end[1] - 1
                        ty = end[0] - rng.randint(0, 1)
                    tgt = [(ty + a, tx + b) for a in range(2) for b in range(2)]
                    allc = block + [tip] + ray + tgt
                    if not free(occ, allc, h, w):
                        continue
                    occ |= halo(allc)
                    for y, x in block + [tip]:
                        g[y][x] = out[y][x] = col
                    for y, x in ray:
                        out[y][x] = col
                    for y, x in tgt:
                        g[y][x] = T
                        out[y][x] = col
                    done += 1
                    break
            if done < 1 or (done < 2 and rng.random() < 0.7):
                continue
            for _ in range(rng.randint(0, 2)):
                for _ in range(50):
                    ty, tx = rng.randint(0, h - 2), rng.randint(0, w - 2)
                    tgt = [(ty + a, tx + b) for a in range(2) for b in range(2)]
                    if free(occ, tgt, h, w):
                        occ |= halo(tgt)
                        for y, x in tgt:
                            g[y][x] = out[y][x] = T
                        break
            # distractor targets must not sit on any block's line of sight: rays are reserved above,
            # and a distractor is never the first thing hit since the real target is closer
            return g, out

    pairs = []
    for _ in range(rng.randint(3, 4) + 1):
        i, o = make()
        pairs.append({"input": i, "output": o})
    return {"train": pairs[:-1], "test": pairs[-1:]}
