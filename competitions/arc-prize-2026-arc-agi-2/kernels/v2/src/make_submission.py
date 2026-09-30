import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

comp_dir = os.environ.get("ARC_COMP_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
if rerun_mode:
    data = ArcDataset.from_file(f"{comp_dir}/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file(f"{comp_dir}/arc-agi_evaluation_challenges.json")
    data = data.load_replies(f"{comp_dir}/arc-agi_evaluation_solutions.json")

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

decoder.load_decoded_results("/kaggle/inference_outputs")

submission = data.get_submission(decoder.run_selection_algo())

# Outputs with no candidate at all: try the test input itself as attempt_1 instead of [[0]].
n_empty = 0
for key, outputs in submission.items():
    for i, attempts in enumerate(outputs):
        if attempts["attempt_1"] == [[0]]:
            attempts["attempt_1"] = data.queries[key]["test"][i]["input"]
            n_empty += 1
print(f"*** outputs without candidates: {n_empty} / {sum(map(len, submission.values()))}")

with open("submission.json", "w") as f:
    json.dump(submission, f)

if not rerun_mode:
    decoder.benchmark_selection_algos()

    # Paired check of later passes: same tasks, pass-1 candidates only vs all passes pooled.
    import re, shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name in os.listdir("/kaggle/inference_outputs"):
            if not re.fullmatch(r"p\d+", name.rsplit(".", 1)[-1]):
                shutil.copy(os.path.join("/kaggle/inference_outputs", name), tmp)
        pass1 = ArcDecoder(data.split_multi_replies(), n_guesses=2)
        pass1.load_decoded_results(tmp)
        pass1_submission = data.get_submission(pass1.run_selection_algo())
    print("*** Pass-1-only score:", data.validate_submission(pass1_submission))
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))

    # Keep the raw candidates for offline selection/re-scoring work (eval commit runs only).
    import tarfile
    with tarfile.open("/kaggle/working/inference_outputs.tar", "w") as tar:
        tar.add("/kaggle/inference_outputs", arcname="inference_outputs")
    print("*** saved inference_outputs.tar:", os.path.getsize("/kaggle/working/inference_outputs.tar") // 1024, "KB")
