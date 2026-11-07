"""verify_all.sh 출력 폴더의 JSON 으로 과제별 배율과 기하평균을 계산한다.
Usage: uv run python src/score.py <out_dir> [<out_dir> ...]"""
import json, math, sys
from pathlib import Path

REF = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155365}

for d in sys.argv[1:]:
    ratios = {}
    for t, ref in REF.items():
        p = Path(d) / f"{t}.json"
        try:
            r = json.loads(p.read_text())
        except Exception as e:
            print(f"{d} {t}: 결과 없음 ({e.__class__.__name__})"); continue
        if r.get("status") != "PASS":
            print(f"{d} {t}: FAIL {r.get('error')}"); continue
        ratios[t] = ref / r["score"]
        print(f"{d} {t}: Q={r.get('num_qubits')} T={r.get('toffoli_count')} cost={r['score']:.6g} ratio={ratios[t]:.4f}")
    if len(ratios) == 3:
        print(f"{d} GM = {math.prod(ratios.values()) ** (1/3):.5f}")
