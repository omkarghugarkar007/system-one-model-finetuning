"""Calibration: temperature per bucket, and conformal intervals on the utility.

Separate from training on purpose. A temperature fitted on the training set is
fitted to memorised answers, and the frontier needs interval coverage, which is
a different property from ECE.
"""
from .conformal import ConformalIntervals, CoverageMonitor, coverage_by_group
from .temperature import TemperatureMap, ece_by_bucket, fit_temperature, reliability_curve

__all__ = ["TemperatureMap", "fit_temperature", "ece_by_bucket",
           "reliability_curve", "ConformalIntervals", "coverage_by_group",
           "CoverageMonitor"]
