"""Protocol token and robust scaling (fitted on training rows only)."""
import numpy as np

# Token per 'Protocol Type' value: 0 (ARP/none) -> 0, ICMP 1 -> 1, TCP 6 -> 2, UDP 17 -> 3, anything else -> 4.
# CICIoMT2024 averages Protocol Type over the packets of a flow, so non-integer values fall into "other".
_PROTO_MAP = ((0.0, 0), (1.0, 1), (6.0, 2), (17.0, 3))
N_PROTO_TOKENS = 5


def protocol_tokens(values):
    v = np.asarray(values, dtype=np.float64)
    tok = np.full(v.shape, N_PROTO_TOKENS - 1, dtype=np.int64)
    for value, token in _PROTO_MAP:
        tok[np.isclose(v, value)] = token
    return tok


class RobustScaler:
    """(x - median) / IQR, clipped to [-clip, clip]. A constant feature gets scale 1."""

    def __init__(self, clip=10.0):
        self.clip = float(clip)
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.center_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        scale = q75 - q25
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float32) - self.center_.astype(np.float32)) / self.scale_.astype(np.float32)
        np.clip(Z, -self.clip, self.clip, out=Z)
        return Z


class StandardScaler(RobustScaler):
    """(x - mean) / std, clipped. Selected with data.scaler: standard."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.center_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self


def build_scaler(kind, clip):
    if kind == "standard":
        return StandardScaler(clip)
    if kind == "robust":
        return RobustScaler(clip)
    raise ValueError(f"data.scaler must be 'robust' or 'standard', got '{kind}'")
