"""Pseudonimisasi pengenal dengan HMAC-SHA256 (Subbab 3.4.1)."""
import hashlib
import hmac
import os

import numpy as np
import pandas as pd


def get_key() -> bytes:
    k = os.environ.get("PGRP_HMAC_KEY")
    if not k:
        raise RuntimeError(
            "Variabel lingkungan PGRP_HMAC_KEY belum diatur. Contoh (PowerShell): "
            "$env:PGRP_HMAC_KEY='kunci-rahasia-anda'. Simpan kunci di luar repositori."
        )
    return k.encode()


def pseudonymize_codes(s: pd.Series, key: bytes) -> np.ndarray:
    """Kode bilangan bulat stabil untuk tiap nilai unik (-1 untuk nilai kosong).

    Kode adalah peringkat digest HMAC-SHA256, sehingga tidak bergantung pada urutan data
    dan nilai mentah tidak pernah disimpan.
    """
    codes, uniques = pd.factorize(s)
    digests = np.array([hmac.new(key, str(u).encode(), hashlib.sha256).hexdigest() for u in uniques])
    order = np.argsort(digests)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    safe = np.where(codes >= 0, codes, 0)
    out = np.where(codes >= 0, rank[safe] if len(rank) else 0, -1)
    return out.astype(np.int64)
