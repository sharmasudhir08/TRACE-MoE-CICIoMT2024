"""Digital Twin core (trace_iomt.twin) on small synthetic tables."""
import copy

import numpy as np
import pandas as pd
import pytest

from trace_iomt import twin
from trace_iomt.pcapfeat import FEATURES


def table(n, device, state, base=10.0, noise=1.0, peers="10.0.0.1", topics="", seed=0, file="f"):
    rng = np.random.default_rng(seed)
    x = np.abs(base + noise * rng.standard_normal((n, len(FEATURES))))
    df = pd.DataFrame(x, columns=FEATURES)
    df["device"], df["state"], df["peers"], df["topics"], df["file"] = device, state, peers, topics, file
    df["t_start"] = np.arange(n, dtype=float)
    return df


def test_feature_matrix_is_log1p_with_nan_as_zero():
    df = table(2, "a", "Idle")
    df.loc[0, "Std"] = np.nan
    x = twin.feature_matrix(df)
    assert x[0, FEATURES.index("Std")] == 0.0
    assert x[1, 0] == pytest.approx(np.log1p(df.iloc[1, 0]))


def test_profiles_fall_back_from_state_to_device_to_fleet():
    rows = pd.concat([table(300, "a", "Idle"), table(50, "a", "Power", seed=1), table(300, "other", "Idle", seed=2)])
    prof = twin.fit_profiles(rows, twinned={"a"}, min_support=200)
    assert prof.lookup("a", "Idle")[1] == "state"
    assert prof.lookup("a", "Power")[1] == "device"          # 50 rows < 200: device-level profile
    assert prof.lookup("other", "Idle")[1] == "fleet"        # not twinned
    assert prof.lookup("a", "Broker")[1] == "device"         # Broker aliases to Active, which a lacks
    assert prof.fallbacks == {"device": 2, "fleet": 1}


def test_shrinkage_gives_constant_features_a_positive_sigma():
    rows = pd.concat([table(300, "a", "Idle", noise=0.0), table(300, "b", "Idle", noise=2.0, seed=1)])
    raw = twin.fit_profiles(rows, twinned={"a"}, shrinkage=0)
    shrunk = twin.fit_profiles(rows, twinned={"a"}, shrinkage=50)
    assert np.all(raw.profiles[("a", "Idle")].sigma < 1e-12)          # constant: σ effectively 0
    assert np.all(shrunk.profiles[("a", "Idle")].sigma > 0.01)        # borrowed from the fleet
    big = twin.fit_profiles(pd.concat([table(100000, "a", "Idle", noise=0.5), rows]), {"a"}, shrinkage=50)
    np.testing.assert_allclose(big.profiles[("a", "Idle")].mu, np.log1p(10.0), atol=0.05)   # n >> k: own values


def test_discrepancy_matches_equation_1():
    p = twin.Profile(mu=np.zeros(2), sigma=np.array([1.0, 2.0]), n=10)
    x = np.array([[1.0, 2.0], [-3.0, 0.0]])
    np.testing.assert_allclose(twin.discrepancy(x, p), [(1 + 3) / 2, (1 + 0) / 2], rtol=1e-5)


def test_novelty_counts_new_peers_and_unparsable_topics():
    p = twin.Profile(np.zeros(1), np.ones(1), 1, peers=frozenset({"10.0.0.1"}), topics=frozenset({"spo2"}))
    assert twin.novelty(["10.0.0.1", "10.0.0.1;10.0.0.9"], ["spo2", ""], p) == (0.5, 0.0)
    assert twin.novelty(["10.0.0.1"], ["?"], p) == (0.0, 1.0)


def test_synchroniser_applies_guard_and_ema():
    rows = table(300, "a", "Idle", base=0.0, noise=0.0)
    prof = twin.fit_profiles(rows, {"a"}, shrinkage=0)
    p = prof.profiles[("a", "Idle")]
    p.sigma[:] = 1.0
    sync = twin.Synchroniser(copy.deepcopy(prof), eta=0.1, delta=5.0, budget_per_hour=2)
    x = np.ones((16, len(FEATURES)))
    small, large = np.full(len(FEATURES), 1.0), np.full(len(FEATURES), 9.0)
    assert not sync.maybe_update("a", "Idle", x, small, confident_benign=False, t=0)
    assert not sync.maybe_update("a", "Idle", x, large, confident_benign=True, t=0)
    assert not sync.maybe_update("zz", "Idle", x, small, confident_benign=True, t=0)
    assert sync.maybe_update("a", "Idle", x, small, True, t=0) and sync.maybe_update("a", "Idle", x, small, True, t=10)
    assert not sync.maybe_update("a", "Idle", x, small, True, t=20)            # budget of 2 per hour used up
    assert sync.maybe_update("a", "Idle", x, small, True, t=3601)              # first update left the window
    new = sync.twin.profiles[("a", "Idle")]
    assert new.mu[0] == pytest.approx(1 - 0.9 ** 3)
    assert prof.profiles[("a", "Idle")].mu[0] == 0.0                           # the static copy is unchanged
    assert sync.counts == {"rejected_not_confident": 1, "rejected_drift_bound": 1, "rejected_not_twinned": 1,
                           "updated": 3, "rejected_budget": 1}


def test_state_samples_time_split_never_overlaps():
    rows = pd.concat([table(100, "a", "Idle", file="x"), table(60, "a", "Active", base=30, seed=1, file="y")])
    X, y, g, starts, lengths = twin.state_samples(rows, w=16, stride=8)
    train, test = twin.holdout_by_time(starts, lengths, w=16)
    assert train.any() and test.any() and not (train & test).any()
    cut = np.floor(0.7 * lengths)
    assert np.all(starts[train] + 16 <= cut[train]) and np.all(starts[test] >= cut[test])


def test_state_estimator_separates_distinct_states():
    rows = pd.concat([table(400, "a", "Idle", base=5, file="x"), table(400, "a", "Active", base=50, seed=1, file="y")])
    X, y, g, starts, lengths = twin.state_samples(rows)
    est = twin.StateEstimator().fit(X, y)
    assert est.predict_rows(twin.feature_matrix(table(16, "a", "?", base=50, seed=3))) == "Active"
