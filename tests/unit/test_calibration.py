"""Calibration invariants. These guard claims the frontier depends on."""
import numpy as np
import pytest

from frontierrank.calibration import (ConformalIntervals, CoverageMonitor,
                                      TemperatureMap, coverage_by_group,
                                      ece_by_bucket, fit_temperature)


def _sharp_records(n=3000, true_t=3.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.choice([2, 4, 8]))
        z = rng.normal(0, 1, k)
        p = np.exp(z) / np.exp(z).sum()
        out.append((z * true_t, int(rng.choice(k, p=p)), "choice"))
    return out


def test_temperature_recovers_a_known_sharpening():
    recs = _sharp_records(true_t=3.0)
    m = TemperatureMap.fit(recs, min_samples=50)
    assert 2.5 < m.global_t < 3.6
    for t in m.temperatures.values():
        assert 2.2 < t < 4.0


def test_temperature_refit_reduces_ece():
    recs = _sharp_records()
    m = TemperatureMap.fit(recs, min_samples=50)
    before = ece_by_bucket(recs)["__aggregate__"]["ece"]
    after = ece_by_bucket(recs, m)["__aggregate__"]["ece"]
    assert after < before / 3


def test_buckets_follow_layas_own_boundaries():
    # the cliff is at 11 options, not 20 -- Laya ships choice:11+ -> T=0.1006
    assert TemperatureMap.bucket("choice", 10) == "choice:6-10"
    assert TemperatureMap.bucket("choice", 11) == "choice:11+"
    assert TemperatureMap.bucket("noul", 2) == "noul:2"


def test_thin_buckets_fall_back_instead_of_fitting_noise():
    recs = _sharp_records(n=400)
    recs += [(np.array([0.1, 0.2, 0.3] + [0.0] * 12), 1, "choice")] * 5
    m = TemperatureMap.fit(recs, min_samples=50)
    assert "choice:11+" in m.fallbacks
    assert m.get("choice", 15) == m.global_t


def test_aggregate_ece_can_hide_a_bad_bucket():
    """Laya's own eval reports 0.030 aggregate hiding a family at 0.438.
    The reporting must make that visible rather than average it away."""
    good = _sharp_records(n=3000, true_t=1.0, seed=1)
    bad = [(z * 8.0, y, "noul") for z, y, _ in _sharp_records(n=120, seed=2)]
    out = ece_by_bucket(good + bad)
    agg = out["__aggregate__"]
    assert agg["worst_bucket_ece"] > agg["ece"]
    assert agg["hiding_ratio"] > 1.5


# ------------------------------------------------------------------ conformal
def test_conformal_widens_sigmas_that_are_too_narrow():
    rng = np.random.default_rng(0)
    n = 2000
    u = rng.normal(0, 1.5, n)
    sigma = np.full(n, 0.3)
    u_hat = u + rng.normal(0, 0.6, n)          #真 error is 2x the claimed sigma
    ci = ConformalIntervals.fit(u, u_hat, sigma, alpha=0.1)
    assert ci.q > 1.64, "should detect that the claimed sigmas are too narrow"


def test_conformal_achieves_its_target_coverage_out_of_sample():
    rng = np.random.default_rng(1)
    n = 3000
    u = rng.normal(0, 1.5, n)
    sigma = np.abs(rng.normal(0.4, 0.1, n)) + 0.05
    u_hat = u + rng.normal(0, 1, n) * sigma * 1.7
    ci = ConformalIntervals.fit(u[:1500], u_hat[:1500], sigma[:1500], alpha=0.1)
    cov = float(ci.covers(u[1500:], u_hat[1500:], sigma[1500:]).mean())
    assert 0.86 < cov < 0.94, f"coverage {cov} missed the 90% target"


def test_conformal_refuses_a_calibration_set_too_small_to_mean_anything():
    with pytest.raises(ValueError, match="too few"):
        ConformalIntervals.fit(np.zeros(10), np.zeros(10), np.ones(10))


def test_coverage_by_group_surfaces_a_badly_covered_slice():
    rng = np.random.default_rng(2)
    n = 2000
    u = rng.normal(0, 1, n)
    sigma = np.full(n, 0.5)
    err = rng.normal(0, 0.5, n)
    err[n // 2:] *= 4.0                       # second group is much harder
    ci = ConformalIntervals.fit(u, u + err, sigma, alpha=0.1)
    groups = np.repeat(["easy", "hard"], n // 2)
    out = coverage_by_group(ci, u, u + err, sigma, groups)
    assert out["hard"]["coverage"] < out["easy"]["coverage"]
    assert out["__overall__"]["worst_group_coverage"] == out["hard"]["coverage"]


def test_monitor_flags_drift():
    rng = np.random.default_rng(3)
    n = 1000
    u = rng.normal(0, 1, n)
    sigma = np.full(n, 0.4)
    ci = ConformalIntervals.fit(u, u + rng.normal(0, 0.4, n), sigma, alpha=0.1)
    mon = CoverageMonitor(ci)
    mon.update(u, u + rng.normal(0, 0.4, n), sigma)
    assert not mon.drifted()
    mon.hits.clear()
    mon.update(u, u + rng.normal(0, 3.0, n), sigma)   # distribution shift
    assert mon.drifted()
