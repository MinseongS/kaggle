import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

comp_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
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
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))
