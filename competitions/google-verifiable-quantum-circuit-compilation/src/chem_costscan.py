"""dfthcsos 비용 모델(heuristic 부분)만으로 (R,B,C,brot) 를 스캔한다. angle 회로 비용은 공개 제출의 per-address 비율로 근사."""
import math, sys
sys.path.insert(0, "data/verifiers/dfthcsos")
from scoreinfra.costs_prepare import SpinFreeData_V1

def heur(R, B, C, brot, N=76, chi=10):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=chi, beta=brot)
    o, i = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    T = m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()
    return T, o["persistent"] + i["persistent"], max(o["temporary"], i["temporary"])

# 공개 best: R9 B143 C39 brot8 J5, angles 8872 Toffoli (1363 addr), angle temp 77
def total(R, B, C, brot, J=5, angle_tof_per_addr=8872 / 1363 * 8 / 8, angle_temp=77):
    T, P, tmp = heur(R, B, C, brot)
    addr = 76 + R * B
    Ta = angle_tof_per_addr * addr * (brot / 8)  # 대략 brot 에 비례한다고 가정
    return T + Ta, P + J * brot + max(tmp, angle_temp), T, P, tmp

base = total(9, 143, 39, 8)
print("public shape model: T=%.0f Q=%d (heurT=%d pers=%d temp=%d)" % base)
for brot in range(5, 11):
    t = total(9, 143, 39, brot)
    print(f"brot={brot:2d}  T={t[0]:7.0f} Q={t[1]:4d}  heur_temp={t[4]:4d}  T*Q ratio vs brot8 = {t[0]*t[1]/(base[0]*base[1]):.3f}")
