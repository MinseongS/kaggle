from datetime import date
from pathlib import Path

HEADER = "| date | exp | change | CV | LB | note |\n|---|---|---|---|---|---|\n"


def log_experiment(
    exp: str,
    change: str,
    cv: float | None = None,
    lb: float | None = None,
    note: str = "",
    path: str | Path = "EXPERIMENTS.md",
) -> None:
    """Append one row to the competition's EXPERIMENTS.md table."""
    path = Path(path)
    if not path.exists() or "| date |" not in path.read_text():
        with path.open("a") as f:
            f.write(HEADER)
    fmt = lambda v: "" if v is None else f"{v:.5f}"
    with path.open("a") as f:
        f.write(f"| {date.today()} | {exp} | {change} | {fmt(cv)} | {fmt(lb)} | {note} |\n")
