# Synthetic ARC-AGI-2-style task generators

Purpose: (1) a **leak-free validation set** — the NVARC model was SFT'd on the ARC-AGI-2 evaluation set, so
our eval numbers are inflated and misled us once (v6); tasks from new generators are instances it has never
seen. (2) Optional LoRA continual-FT data from *other* generators.

## Generator contract (`gens/<name>.py`)
```python
CONCEPT = "one line: the rule a solver must infer"
def generate(rng: random.Random) -> dict:
    # returns {"train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid, "output": grid}]}
```
- grid = list of lists of int 0..9, rectangular, each side 1..30. Use only `rng` for randomness (deterministic per seed).
- 3–4 train pairs (vary it), exactly 1 test pair. Pairs within a task must follow ONE hidden rule; the rule's
  parameters (which colour, which direction, which shape...) are drawn per task, so different tasks differ.
- The rule must be **inferable from the train pairs** (every parameter is exposed by at least one train pair)
  and the test output must be unique given the rule.
- ARC-AGI-2 flavour, not ARC-1 toy: prefer compositional rules (2–3 interacting steps), rules conditioned on
  context (object property, a key/legend in the grid, counting), objects of several colours/shapes, grids
  typically 10x10–20x20, background usually 0 but not always. Avoid trivial rules (pure recolour, flip,
  copy) unless combined with something else.
- input != output for every pair; no two train inputs identical.
- Pure python stdlib only (random, itertools, math, collections). No numpy.
- If a draw is invalid (objects overlap, ambiguity), retry internally — `generate` must always return.

## Validation
`uv run python synth/check.py gens/<name>.py` → generates 200 tasks with seeds 0..199 and asserts the contract
(shapes, colours, counts, input != output, determinism, test output not equal to any train output, ...).
