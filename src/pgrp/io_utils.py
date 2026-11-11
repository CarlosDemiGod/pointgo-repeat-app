"""Baca/tulis tabel: Parquet bila pyarrow tersedia, selain itu pickle (hanya untuk pengujian)."""
from pathlib import Path
import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def _path(stem) -> Path:
    stem = Path(stem)
    return stem.with_suffix(".parquet" if HAS_PARQUET else ".pkl")


def write_table(df: pd.DataFrame, stem) -> Path:
    p = _path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        df.to_parquet(p, index=False)
    else:
        df.to_pickle(p)
    return p


def read_table(stem) -> pd.DataFrame:
    p = _path(stem)
    return pd.read_parquet(p) if HAS_PARQUET else pd.read_pickle(p)


def table_exists(stem) -> bool:
    return _path(stem).exists()
