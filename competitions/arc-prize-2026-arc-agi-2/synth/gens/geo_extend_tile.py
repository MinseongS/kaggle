import random

CONCEPT = ("a small periodic 2-D pattern is shown in one corner of an empty grid; continue the pattern from that "
           "corner until it fills the whole grid")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def _min_period_extend(vis, h, w):
    sh, sw = len(vis), len(vis[0])
    cands = sorted(((a, b) for a in range(1, sh + 1) for b in range(1, sw + 1)), key=lambda x: (x[0] * x[1], x))
    for a, b in cands:
        if all(vis[r][c] == vis[r % a][c % b] for r in range(sh) for c in range(sw)):
            return [[vis[r % a][c % b] for c in range(w)] for r in range(h)]


def generate(rng: random.Random) -> dict:
    def make():
        while True:
            ph, pw = rng.randint(1, 4), rng.randint(2, 4)
            if rng.random() < 0.5:
                ph, pw = pw, ph
            cols = rng.sample(range(1, 10), rng.randint(2, 3))
            tile = [[rng.choice(cols) for _ in range(pw)] for _ in range(ph)]
            if len({v for row in tile for v in row}) < 2:
                continue
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            sh = min(h - 2, 2 * ph + rng.randint(0, 2))
            sw = min(w - 2, 2 * pw + rng.randint(0, 2))
            sh, sw = max(sh, 2), max(sw, 2)
            out = [[tile[r % ph][c % pw] for c in range(w)] for r in range(h)]
            vis = [row[:sw] for row in out[:sh]]
            if _min_period_extend(vis, h, w) != out:
                continue
            inp = [[out[r][c] if r < sh and c < sw else 0 for c in range(w)] for r in range(h)]
            corner = rng.randint(0, 3)
            if corner & 1:
                inp, out = [r[::-1] for r in inp], [r[::-1] for r in out]
            if corner & 2:
                inp, out = inp[::-1], out[::-1]
            return inp, out

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
