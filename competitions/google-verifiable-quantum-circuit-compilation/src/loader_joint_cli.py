"""Usage: uv run python src/loader_joint_cli.py <submission.npz> <k> <plan.pkl> <out.kmb>"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loader_v2 import compile_joint, emit, check, gauge_flags, compile_joint_k0, emit_k0
W = np.load(sys.argv[1])["angle_words"]; k = int(sys.argv[2]); plan = pickle.load(open(sys.argv[3], "rb"))
if k == 0:
    prods, pure = compile_joint_k0(W, plan); c = emit_k0(prods, pure, plan['mono'])
else:
    prods, pure = compile_joint(W, k, plan)
    c = emit(prods, pure, k, gauge_flags(W, k))
print("static toffoli", c.operation_statistics().num_toffoli_like, "qubits", c.num_qubits, "products", len(prods))
ok, tof = check(W, k, c)
print("check", ok, "mean toffoli/pass", tof)
if ok: c.to_file(sys.argv[4], format="kmb")
