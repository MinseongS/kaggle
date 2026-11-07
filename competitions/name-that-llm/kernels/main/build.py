# Builds kernels/main/ntl-main.ipynb from kernels/pub667 + lite2 member (s004).
import json, copy, pathlib
H = pathlib.Path(__file__).resolve().parent
nb = json.load(open(H.parent / 'pub667' / 'ntl-pub667.ipynb'))
MINE_W = 0.25

def src(c): return ''.join(c['source'])
def setsrc(c, s): c['source'] = s.splitlines(keepends=True)

# 1) infer.py hook: geometric mix of our lite2 probabilities after the comp stack, before ATLAS
c = nb['cells'][14]; s = src(c)
a0 = "p.add_argument('--comp_path', default=''); p.add_argument('--comp_w', type=float, default=0.0)\n"
assert s.count(a0) == 1
s = s.replace(a0, a0 + "p.add_argument('--mine_path', default=''); p.add_argument('--mine_w', type=float, default=0.0)\n")
b0 = "if (a.sinkhorn > 0 and len(test) >= 18) or _bundles or COMP_P is not None:\n"
assert s.count(b0) == 1
s = s.replace(b0, """MINE_P = None
if a.mine_w > 0 and a.mine_path and os.path.exists(a.mine_path):   # ntl-lite2 member (own process)
    try:
        _mm = json.load(open(a.mine_path.replace('.npy', '.json')))
        assert _mm['ids'] == test['id'].astype(str).tolist(), 'mine row order differs'
        assert _mm['labels'] == LABELS
        MINE_P = np.load(a.mine_path).astype(np.float64); MINE_P = np.clip(MINE_P, 1e-9, None); MINE_P /= MINE_P.sum(1, keepdims=True)
        assert MINE_P.shape == (len(test), 18) and np.isfinite(MINE_P).all(); log('mine loaded', MINE_P.shape)
    except Exception as _e:
        log('MINE SKIPPED:', repr(_e)); MINE_P = None
elif a.mine_w > 0: log('MINE SKIPPED: no output')
""" + b0)
c0 = "    if COMP_P is not None: P = (1 - a.comp_w) * P + a.comp_w * COMP_P; P /= P.sum(1, keepdims=True)\n"
assert s.count(c0) == 1
s = s.replace(c0, c0 + """    if MINE_P is not None:
        _lm = (1 - a.mine_w) * np.log(P + 1e-12) + a.mine_w * np.log(MINE_P)
        _p2 = np.exp(_lm - _lm.max(1, keepdims=True)); _p2 /= _p2.sum(1, keepdims=True)
        log(f'mine mixed at {a.mine_w}: argmax changed {int((_p2.argmax(1) != P.argmax(1)).sum())}/{len(P)}'); P = _p2
""")
setsrc(c, s)

# 2) lite2 member script + run cell, inserted before the members/run section (cell 19)
mine_py = (H / 'mine.py').read_text()
cw = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': []}
c1 = copy.deepcopy(cw); setsrc(c1, '%%writefile mine.py\n' + mine_py)
c2 = copy.deepcopy(cw); setsrc(c2, """# ntl-lite2 member (TF-IDF SVC on train+pool, Qwen3-1.7B layer head) in its own process; runs on the commit too
import time as _t
_m0 = _t.time()
!python mine.py 2>&1 | grep -vE 'Warning|warn'
MINE_SEC = _t.time() - _m0
print('mine wall time', round(MINE_SEC))
""")
md = {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 5b. Own member: ntl-lite2 probabilities (mixed at %.2f in log space)\n' % MINE_W]}
nb['cells'][19:19] = [md, c1, c2]

# 3) run cell: pass the hook, charge mine time to the ATLAS budget
c = nb['cells'][23]; s = src(c)
d0 = "ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)"
assert s.count(d0) == 1; s = s.replace(d0, "ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - MINE_SEC - 600)")
e0 = "       '--comp_path', '/kaggle/working/comp_logit.npy', '--comp_w', '0.15',\n"
assert s.count(e0) == 1
s = s.replace(e0, e0 + "       '--mine_path', '/kaggle/working/mine_prob.npy', '--mine_w', '%s',\n" % MINE_W)
setsrc(c, s)
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
json.dump(nb, open(H / 'ntl-main.ipynb', 'w'), indent=1)
print('ok', len(nb['cells']))
