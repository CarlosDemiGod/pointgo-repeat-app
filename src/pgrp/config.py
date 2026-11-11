from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str = "configs/base.yaml") -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.parent.parent
    work = Path(cfg["paths"]["work_dir"])
    work = work if work.is_absolute() else root / work
    work.mkdir(parents=True, exist_ok=True)
    cfg["root"], cfg["work"] = root, work
    return cfg


def get_cutoffs(cfg: dict) -> list:
    c = cfg["cutoffs"]
    return list(pd.date_range(c["first"], c["last"], freq="MS"))
