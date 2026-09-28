import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    target: str | None = None,
    group: str | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return a fold id per row. Stratify by `target` and/or keep `group` rows together."""
    y = df[target] if target else None
    groups = df[group] if group else None
    if target and group:
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    elif group:
        splitter = GroupKFold(n_splits)
    elif target:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=seed)

    folds = np.full(len(df), -1, dtype=int)
    for k, (_, valid_idx) in enumerate(splitter.split(df, y, groups)):
        folds[valid_idx] = k
    return folds
