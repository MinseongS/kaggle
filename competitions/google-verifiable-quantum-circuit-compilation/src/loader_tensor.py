"""chem angle loader 의 mixed ANF 텐서 (low 4비트 x high 7비트 x 출력 35비트) 계산 유틸.
도메인 [0,1220) 은 부분집합 순서의 down-set 이라 ANF 계수가 m<1220 에서 잘 정의된다."""
import numpy as np
A = 1220
LOWM = list(range(1, 16))                 # low 단항식 (공집합 제외)
HIGHM = [h for h in range(1, 77)]         # high 단항식 h (주소 비트 4..10), h*16 < 1220
def anf_bits(vals):
    """vals: (A, n) uint8 0/1 -> ANF 계수 (A, n), 길이 A 의 down-set 위 Möbius."""
    out = vals.copy().astype(np.uint8)
    for bit in range(11):
        m = np.arange(A); sel = (m >> bit) & 1 == 1
        out[sel] ^= out[m[sel] ^ (1 << bit)]
    return out
def delta(words, k, J=5):
    d = words[:, k*J:(k+1)*J].copy()
    if k: d ^= words[:, (k-1)*J:k*J]
    return d
def out_bits(d):
    """d (A,5) -> (A,35) bits 0..6 of each word, order word-major"""
    return np.concatenate([((d[:, j:j+1] >> np.arange(7)) & 1) for j in range(d.shape[1])], axis=1).astype(np.uint8)
def mixed_tensor(bits):
    """bits (A,n) -> T (15, 76, n) mixed coeffs, dc (15,76) bool don't-care mask, pure coefficients dict"""
    c = anf_bits(bits)
    n = bits.shape[1]
    T = np.zeros((15, 76, n), np.uint8); dc = np.zeros((15, 76), bool)
    for li, L in enumerate(LOWM):
        for hi, h in enumerate(HIGHM):
            m = h * 16 + L
            if m < A: T[li, hi] = c[m]
            else: dc[li, hi] = True
    return T, dc, c
def gf2rank(M):
    M = (M.copy() & 1).astype(np.uint8); r = 0
    rows, cols = M.shape
    for c in range(cols):
        if r == rows: break
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        p = r + piv[0]
        if p != r: M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]; nz = nz[nz != r]
        M[nz] ^= M[r]; r += 1
    return r
