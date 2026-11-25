r"""Robustness protocols: multi-seed, leave-one-out and leave-one-time-point-out.

Nothing in the methodology is altered: the DGM surrogate, the absorption-driven
residual law, the softplus parameterisation, the collocation density, the
optimiser and the 6000-epoch budget are those of the reference run.  Only the
supervision subset, the RNG seed and (for one labelled control arm) the physics
weight are varied.

Protocols
---------
1. MULTI-SEED
   Configurations C and D retrained under five seeds {0, 42, 123, 2025, 2026}.
   Mean +/- SD of MSE / RMSE / R^2 and of every recovered kinetic constant, plus
   the seed-resolved dense trajectories.

2. LOO (leave-one-observation-out)
   One of the 16 scalar measurements (8 nodes x 2 compartments) is withheld per
   fold; the sibling compartment at that node is retained.  16 folds,
   Configuration D.

3. LOTPO (leave-one-time-point-out)
   The entire node (both compartments) is withheld.  8 folds, for
   Configuration C, Configuration D and a physics-free control (Configuration D
   with lambda_p = 0).

Leakage control
---------------
The initial-condition anchor is built from measured values, so a fold that
withholds an anchored value must not keep the corresponding anchor term:

* LOTPO fold 0 (t = 0.083 h) re-anchors on the earliest retained node
  (t = 0.330 h), imposed at its own normalised time.
* LOO folds at node 0 switch off only the withheld compartment's anchor term.

Every run is appended to ``tables/robustness_raw.jsonl`` as soon as it finishes,
so an interrupted suite resumes where it stopped.  Each worker uses one thread.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .config import C_MAX

SEEDS: Tuple[int, ...] = (0, 42, 123, 2025, 2026)
COMPARTMENTS = ("plasma", "tumour")
DENSE_N = 200


# --------------------------------------------------------------------------- #
# Job specification
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    protocol: str            # "multiseed" | "loo" | "lotpo"
    arm: str                 # "C" | "D" | "D_nophys"
    seed: int
    fold: int                # -1 for multiseed
    held_node: int           # -1 when nothing is withheld
    held_comp: int           # -1 = both compartments withheld / none
    epochs: int

    @property
    def label(self) -> str:
        if self.protocol == "multiseed":
            return f"{self.protocol}:{self.arm}:seed{self.seed}"
        comp = "both" if self.held_comp < 0 else COMPARTMENTS[self.held_comp]
        return f"{self.protocol}:{self.arm}:node{self.held_node}:{comp}"


ARM_SPEC = {
    # arm -> (index into M.ABLATION, physics_weight)
    "C": (2, 1.0),
    "D": (3, 1.0),
    "D_nophys": (3, 0.0),
}


def build_jobs(epochs: int) -> List[Job]:
    jobs: List[Job] = []
    for arm in ("C", "D"):
        for seed in SEEDS:
            jobs.append(Job("multiseed", arm, seed, -1, -1, -1, epochs))
    for arm in ("D", "C", "D_nophys"):
        for node in range(8):
            jobs.append(Job("lotpo", arm, 0, node, node, -1, epochs))
    for node in range(8):
        for comp in (0, 1):
            jobs.append(Job("loo", "D", 0, 2 * node + comp, node, comp, epochs))
    return jobs


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def default_workers() -> int:
    """Conservative worker count: each worker holds a full float64 model and its
    autograd graph, so memory grows with the number of workers."""
    return min(6, max(1, (os.cpu_count() or 4) - 2))


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)


def run_job(job: Job) -> Dict:
    """Train one robustness fold and return its JSON-serialisable record."""
    import torch

    from . import config as M
    from .data import Normaliser
    from .training import train_configuration

    torch.set_num_threads(1)
    norm = Normaliser(M.T_MAX, M.C_MAX)
    target = norm.ingest()
    cfg_idx, physics_weight = ARM_SPEC[job.arm]
    cfg = M.ABLATION[cfg_idx]

    n = target.n
    obs_mask: Optional[torch.Tensor] = None
    bound_node, bound_at_zero, bound_mask = 0, True, (True, True)

    if job.protocol == "lotpo":
        obs_mask = torch.ones((n, 2), dtype=M.DTYPE, device=M.DEVICE)
        obs_mask[job.held_node, :] = 0.0
        if job.held_node == 0:
            # earliest retained node supplies the anchor, at its own time
            bound_node, bound_at_zero = 1, False
    elif job.protocol == "loo":
        obs_mask = torch.ones((n, 2), dtype=M.DTYPE, device=M.DEVICE)
        obs_mask[job.held_node, job.held_comp] = 0.0
        if job.held_node == 0:
            bound_mask = (job.held_comp != 0, job.held_comp != 1)

    t0 = time.time()
    res = train_configuration(
        cfg, target, norm,
        epochs=job.epochs, n_collocation=300, log_every=0,
        seed=job.seed, obs_mask=obs_mask,
        bound_node=bound_node, bound_at_zero=bound_at_zero,
        bound_mask=bound_mask, physics_weight=physics_weight,
    )
    elapsed = time.time() - t0

    res.surrogate.eval()
    with torch.no_grad():
        pred_nodes = res.surrogate(target.t_bar)
        cp_hat = norm.denorm_conc(pred_nodes[:, 0]).tolist()
        ct_hat = norm.denorm_conc(pred_nodes[:, 1]).tolist()
        t_dense = torch.linspace(0.0, 1.0, DENSE_N, dtype=M.DTYPE,
                                 device=M.DEVICE).reshape(-1, 1)
        pd = res.surrogate(t_dense)
        dense = [
            [float(t_dense[i, 0]) * M.T_MAX,
             float(norm.denorm_conc(pd[i, 0])),
             float(norm.denorm_conc(pd[i, 1]))]
            for i in range(DENSE_N)
        ]

    out = {
        "job": asdict(job),
        "label": job.label,
        "elapsed_s": elapsed,
        "metrics": res.metrics,                 # in-sample over all 8 nodes
        "params": res.final_rates,
        "lambdas": res.lambdas,
        "final_loss": res.loss_history[-1],
        "pred_nodes": {"Cp": cp_hat, "Ct": ct_hat},
        "dense": dense,
        "t_hours": target.t_hours.flatten().tolist(),
        "obs": {"Cp": target.cp_raw.flatten().tolist(),
                "Ct": target.ct_raw.flatten().tolist()},
    }
    return out


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def mean_sd(xs: List[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)   # sample SD (n-1)
    return m, math.sqrt(var)


def r2_against(obs: List[float], pred: List[float]) -> float:
    mu = sum(obs) / len(obs)
    ss_res = sum((p - o) ** 2 for p, o in zip(pred, obs))
    ss_tot = sum((o - mu) ** 2 for o in obs)
    return 1.0 - ss_res / ss_tot


def rmse(obs: List[float], pred: List[float]) -> float:
    return math.sqrt(sum((p - o) ** 2 for p, o in zip(pred, obs)) / len(obs))


def mae(obs: List[float], pred: List[float]) -> float:
    return sum(abs(p - o) for p, o in zip(pred, obs)) / len(obs)


def paired_t(a: List[float], b: List[float]) -> Tuple[float, float, int]:
    """Paired two-sided t statistic, p-value and degrees of freedom."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m, sd = mean_sd(d)
    if sd == 0.0:
        return float("inf") if m != 0 else 0.0, 0.0, n - 1
    t = m / (sd / math.sqrt(n))
    df = n - 1
    # two-sided p from the Student-t CDF via the regularised incomplete beta
    x = df / (df + t * t)
    p = _betainc(0.5 * df, 0.5, x)
    return t, p, df


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (NR 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * \
        _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-14) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_robustness(
    output_dir: str = "outputs",
    epochs: int = 6000,
    workers: Optional[int] = None,
    aggregate_only: bool = False,
) -> Dict:
    """Run (or resume) the robustness suite and write its tables.

    Returns the summary dictionary also written to ``robustness_summary.json``.
    """
    if workers is None:
        workers = default_workers()
    tables = os.path.join(output_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    ckpt_path = os.path.join(tables, "robustness_raw.jsonl")

    records: List[Dict] = []
    done: set = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["job"]["epochs"] != epochs:
                    raise ValueError(
                        f"{ckpt_path} holds runs trained for {rec['job']['epochs']} epochs, "
                        f"but {epochs} were requested; use another output directory.")
                if rec["label"] not in done:
                    records.append(rec)
                    done.add(rec["label"])
        print(f"Checkpoint: {len(records)} completed runs recovered from {ckpt_path}")

    jobs = [j for j in build_jobs(epochs) if j.label not in done]
    if jobs and not aggregate_only:
        print(f"Dispatching {len(jobs)} training runs on {workers} workers "
              f"({epochs} epochs each)", flush=True)
        import multiprocessing as mp
        t_start = time.time()
        total = len(jobs)
        with open(ckpt_path, "a") as ck:
            with mp.Pool(processes=workers, initializer=_init_worker) as pool:
                for i, rec in enumerate(pool.imap_unordered(run_job, jobs), start=1):
                    records.append(rec)
                    ck.write(json.dumps(rec) + "\n")
                    ck.flush()
                    os.fsync(ck.fileno())
                    print(f"  [{i:02d}/{total}] {rec['label']:<42s} "
                          f"{rec['elapsed_s']:6.1f}s", flush=True)
        print(f"All runs finished in {time.time() - t_start:.1f}s wall clock", flush=True)
    elif jobs:
        print(f"--aggregate-only: {len(jobs)} runs are still missing; "
              f"aggregating the {len(records)} that are present.", flush=True)
    else:
        print("All runs already present in the checkpoint; aggregating only.")
    if not records:
        raise RuntimeError("no robustness runs available to aggregate")

    by_label = {r["label"]: r for r in records}
    t_hours = records[0]["t_hours"]
    obs_cp = records[0]["obs"]["Cp"]
    obs_ct = records[0]["obs"]["Ct"]
    summary: Dict = {"epochs": epochs, "seeds": list(SEEDS)}
    summary["completed_runs"] = sorted(by_label)

    def have(*labels: str) -> bool:
        return all(lbl in by_label for lbl in labels)

    ms_arms = [a for a in ("C", "D")
               if have(*[f"multiseed:{a}:seed{sd}" for sd in SEEDS])]
    lotpo_arms = [a for a in ("D", "C", "D_nophys")
                  if have(*[f"lotpo:{a}:node{n}:both" for n in range(8)])]
    loo_ready = have(*[f"loo:D:node{n}:{c}" for n in range(8) for c in COMPARTMENTS])
    print(f"Aggregating: multiseed={ms_arms or 'none'}  "
          f"lotpo={lotpo_arms or 'none'}  loo={'D' if loo_ready else 'none'}")

    # ---------------- multi-seed ------------------------------------------- #
    with open(os.path.join(tables, "multiseed_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "compartment", "MSE", "RMSE", "R2", "mean_R2"])
        for arm in ms_arms:
            for seed in SEEDS:
                r = by_label[f"multiseed:{arm}:seed{seed}"]
                mr2 = 0.5 * (r["metrics"]["plasma"]["R2"] + r["metrics"]["tumour"]["R2"])
                for comp in COMPARTMENTS:
                    m = r["metrics"][comp]
                    w.writerow([arm, seed, comp, m["MSE"], m["RMSE"], m["R2"], mr2])

    with open(os.path.join(tables, "multiseed_parameters.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "k10", "k_in", "k_out", "k_a", "C0_norm", "C0_ugml"])
        for arm in ms_arms:
            for seed in SEEDS:
                p = by_label[f"multiseed:{arm}:seed{seed}"]["params"]
                w.writerow([arm, seed, p["k10"], p["k_in"], p["k_out"], p["k_a"],
                            p["C0"], p["C0"] * C_MAX])

    with open(os.path.join(tables, "multiseed_trajectories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "t_hours", "Cp_pred_ugml", "Ct_pred_ugml"])
        for arm in ms_arms:
            for seed in SEEDS:
                for row in by_label[f"multiseed:{arm}:seed{seed}"]["dense"]:
                    w.writerow([arm, seed] + row)

    ms: Dict = {}
    for arm in ms_arms:
        recs = [by_label[f"multiseed:{arm}:seed{s}"] for s in SEEDS]
        entry: Dict = {}
        for comp in COMPARTMENTS:
            for key in ("MSE", "RMSE", "R2"):
                m, sd = mean_sd([r["metrics"][comp][key] for r in recs])
                entry[f"{comp}_{key}"] = [m, sd]
        m, sd = mean_sd([0.5 * (r["metrics"]["plasma"]["R2"] + r["metrics"]["tumour"]["R2"])
                         for r in recs])
        entry["mean_R2"] = [m, sd]
        for pkey in ("k10", "k_in", "k_out", "k_a", "C0"):
            m, sd = mean_sd([r["params"][pkey] for r in recs])
            entry[pkey] = [m, sd]
            entry[f"{pkey}_cv_pct"] = 100.0 * sd / m if m else 0.0
        ms[arm] = entry
    summary["multiseed"] = ms

    if "C" in ms_arms and "D" in ms_arms:
        a = [0.5 * (by_label[f"multiseed:D:seed{sd}"]["metrics"]["plasma"]["R2"]
                    + by_label[f"multiseed:D:seed{sd}"]["metrics"]["tumour"]["R2"])
             for sd in SEEDS]
        b = [0.5 * (by_label[f"multiseed:C:seed{sd}"]["metrics"]["plasma"]["R2"]
                    + by_label[f"multiseed:C:seed{sd}"]["metrics"]["tumour"]["R2"])
             for sd in SEEDS]
        t_stat, p_val, df = paired_t(a, b)
        diff_m, diff_sd = mean_sd([x - y for x, y in zip(a, b)])
        summary["C_vs_D_paired"] = {
            "mean_diff_D_minus_C": diff_m, "sd_diff": diff_sd,
            "t": t_stat, "p": p_val, "df": df,
            "single_seed_gap_seed0": a[0] - b[0],
        }

    # ---------------- LOO / LOTPO ------------------------------------------ #
    rows: List[List] = []
    for node in (range(8) if loo_ready else ()):
        for comp in (0, 1):
            r = by_label[f"loo:D:node{node}:{COMPARTMENTS[comp]}"]
            key = "Cp" if comp == 0 else "Ct"
            o = (obs_cp if comp == 0 else obs_ct)[node]
            p = r["pred_nodes"][key][node]
            rows.append(["LOO", "D", node, t_hours[node], COMPARTMENTS[comp],
                         o, p, p - o, 100.0 * abs(p - o) / o])
    for arm in lotpo_arms:
        for node in range(8):
            r = by_label[f"lotpo:{arm}:node{node}:both"]
            for comp, key, obs in ((0, "Cp", obs_cp), (1, "Ct", obs_ct)):
                o, p = obs[node], r["pred_nodes"][key][node]
                rows.append(["LOTPO", arm, node, t_hours[node], COMPARTMENTS[comp],
                             o, p, p - o, 100.0 * abs(p - o) / o])
    with open(os.path.join(tables, "loocv_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "config", "held_node", "t_hours", "compartment",
                    "observed", "predicted", "residual", "abs_pct_error"])
        w.writerows(rows)

    with open(os.path.join(tables, "lotpo_trajectories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "held_node", "t_hours", "Cp_pred_ugml", "Ct_pred_ugml"])
        for arm in lotpo_arms:
            for node in range(8):
                for row in by_label[f"lotpo:{arm}:node{node}:both"]["dense"]:
                    w.writerow([arm, node] + row)

    cv: Dict = {}
    cv_targets = ([("LOO", "D")] if loo_ready else []) + [("LOTPO", a) for a in lotpo_arms]
    for proto, arm in cv_targets:
        entry: Dict = {}
        pooled_o, pooled_p = [], []
        for comp, obs in (("plasma", obs_cp), ("tumour", obs_ct)):
            sel = [r for r in rows if r[0] == proto and r[1] == arm and r[4] == comp]
            sel.sort(key=lambda r: r[2])
            o = [r[5] for r in sel]
            p = [r[6] for r in sel]
            entry[comp] = {
                "RMSE": rmse(o, p), "MAE": mae(o, p), "R2": r2_against(o, p),
                "MSE": rmse(o, p) ** 2,
                "max_abs_err": max(abs(x - y) for x, y in zip(o, p)),
                "mean_abs_pct": sum(r[8] for r in sel) / len(sel),
            }
            pooled_o += o
            pooled_p += p
        entry["pooled"] = {
            "RMSE": rmse(pooled_o, pooled_p),
            "MAE": mae(pooled_o, pooled_p),
            "R2": r2_against(pooled_o, pooled_p),
        }
        entry["mean_R2"] = 0.5 * (entry["plasma"]["R2"] + entry["tumour"]["R2"])
        # Interior folds only (nodes 1-6).  Withholding node 0 or node 7 removes a
        # domain endpoint and converts the query from interpolation to
        # extrapolation, so the two regimes are reported separately rather than
        # averaged into a single headline number.
        interior: Dict = {}
        for comp in COMPARTMENTS:
            sel = [r for r in rows if r[0] == proto and r[1] == arm
                   and r[4] == comp and 0 < r[2] < 7]
            sel.sort(key=lambda r: r[2])
            o = [r[5] for r in sel]
            pr = [r[6] for r in sel]
            interior[comp] = {"RMSE": rmse(o, pr), "MAE": mae(o, pr),
                              "R2": r2_against(o, pr),
                              "mean_abs_pct": sum(r[8] for r in sel) / len(sel)}
        interior["mean_R2"] = 0.5 * (interior["plasma"]["R2"] + interior["tumour"]["R2"])
        entry["interior"] = interior
        cv[f"{proto}_{arm}"] = entry
    summary["cross_validation"] = cv

    # in-sample reference for the same arms (seed 0, all 8 nodes supervised)
    summary["in_sample_seed0"] = {
        arm: by_label[f"multiseed:{arm}:seed0"]["metrics"] for arm in ms_arms
    }

    with open(os.path.join(tables, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------- console report --------------------------------------- #
    print("\n" + "=" * 78)
    print("MULTI-SEED SUMMARY  (mean +/- SD over seeds %s)" % (SEEDS,))
    print("=" * 78)
    for arm in ms_arms:
        e = ms[arm]
        print(f"Configuration {arm}")
        for comp in COMPARTMENTS:
            print(f"  {comp:7s} R2={e[comp+'_R2'][0]:.4f}+/-{e[comp+'_R2'][1]:.4f}  "
                  f"RMSE={e[comp+'_RMSE'][0]:7.3f}+/-{e[comp+'_RMSE'][1]:.3f}  "
                  f"MSE={e[comp+'_MSE'][0]:9.2f}+/-{e[comp+'_MSE'][1]:.2f}")
        print(f"  mean R2 = {e['mean_R2'][0]:.4f} +/- {e['mean_R2'][1]:.4f}")
        for pkey in ("k10", "k_in", "k_out", "k_a", "C0"):
            print(f"  {pkey:5s} = {e[pkey][0]:.4f} +/- {e[pkey][1]:.4f}   "
                  f"(CV {e[pkey+'_cv_pct']:.2f}%)")
    d = summary.get("C_vs_D_paired")
    if d:
        print(f"\nPaired D-C mean R2 difference = {d['mean_diff_D_minus_C']:+.4f} "
              f"+/- {d['sd_diff']:.4f}  (t={d['t']:.3f}, df={d['df']}, p={d['p']:.4f})")
        print("Single-seed (seed 0) gap = "
              f"{d['single_seed_gap_seed0']:+.4f}")

    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE CROSS-VALIDATION")
    print("=" * 78)
    for key, e in cv.items():
        print(f"{key}")
        for comp in COMPARTMENTS:
            c = e[comp]
            print(f"  {comp:7s} RMSE={c['RMSE']:8.3f}  MAE={c['MAE']:8.3f}  "
                  f"R2={c['R2']:8.4f}  max|e|={c['max_abs_err']:8.3f}  "
                  f"MAPE={c['mean_abs_pct']:6.2f}%")
        print(f"  pooled  RMSE={e['pooled']['RMSE']:8.3f}  R2={e['pooled']['R2']:.4f}  "
              f"mean R2={e['mean_R2']:.4f}")
        it = e["interior"]
        print(f"  interior folds (nodes 1-6): plasma R2={it['plasma']['R2']:7.4f}  "
              f"tumour R2={it['tumour']['R2']:7.4f}  mean R2={it['mean_R2']:.4f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
