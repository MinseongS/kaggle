"""비용 모델로 (R,B,C,brot) 전체 스캔. angle loader 비용은 공개 제출(1363 주소, brot8, 8872 Toffoli)에 주소 수·brot 비례로 근사.
λ_eff 와 오차 조건은 반영하지 않음 → 후보 형상 탐색용."""
import math, sys, itertools
sys.path.insert(0, "data/verifiers/dfthcsos")
from scoreinfra.costs_prepare import SpinFreeData_V1

def cost(R, B, C, brot, J=5, angle_temp=77):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=76, chi=10, beta=brot)
    try:
        o, i = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
        Th = m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()
    except ValueError:
        return None
    Ta = 8872 * (76 + R * B) / 1363 * brot / 8
    Q = o["persistent"] + i["persistent"] + J * brot + max(o["temporary"], i["temporary"], angle_temp)
    return Th + Ta, Q, Th, max(o["temporary"], i["temporary"])

ref = cost(9, 143, 39, 8)
print("public R9 B143 C39 b8: T=%.0f Q=%d heurT=%d temp=%d" % ref)
rows = []
for R, B, C, brot in itertools.product(range(4, 17), range(64, 257, 4), range(20, 81, 2), [7, 8]):
    c = cost(R, B, C, brot)
    if c is None: continue
    # 표현력 대략 지표: R*B*C (공개 50193) 와 R*B
    rows.append((c[0] * c[1] / (ref[0] * ref[1]), R, B, C, brot, *c))
rows.sort()
print("TQ/pub  R   B   C brot     T    Q  heurT temp  R*B*C")
seen = 0
for r in rows:
    if r[1] * r[2] * r[3] < 40000: continue  # 너무 작은 형상은 오차 조건 못 맞출 것
    print("%.3f %3d %3d %3d %3d %6.0f %4d %6d %4d %6d" % (*r[:5], r[5], r[6], r[7], r[8], r[1]*r[2]*r[3]))
    seen += 1
    if seen > 25: break
