"""Consistency of the reference results shipped in results/."""

import csv
import json
import os

import pytest

from folate_twin.pipeline import reference_metrics


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_reported_metrics_match_package_reference(results_dir):
    ref = reference_metrics()
    rows = _rows(os.path.join(results_dir, "tables", "metric_evaluation_matrix.csv"))
    assert len(rows) == 8
    for r in rows:
        assert ref[(r["config"], r["compartment"])] == r


def test_reported_accuracy(results_dir):
    rows = _rows(os.path.join(results_dir, "tables", "metric_evaluation_matrix.csv"))
    assert min(float(r["R2"]) for r in rows) > 0.96
    d = {r["compartment"]: float(r["R2"]) for r in rows if r["config"] == "Configuration D"}
    assert d["plasma"] > 0.98 and d["tumour"] > 0.98


def test_robustness_summary(results_dir):
    with open(os.path.join(results_dir, "tables", "robustness_summary.json")) as f:
        s = json.load(f)
    assert s["epochs"] == 6000
    assert s["seeds"] == [0, 42, 123, 2025, 2026]
    assert len(s["completed_runs"]) == 50


def test_sensitivity_field_terminal_value(results_dir):
    from folate_twin.viz.figures import DataProvider
    d = DataProvider(results_dir).data
    assert d.dct_dc0 == pytest.approx(0.184, abs=5e-4)
