#!/usr/bin/env python3
"""
13_FORMAL_STATISTICAL_ANALYSIS_V12.py

Formal statistical analysis for the V12 multi-hallmark OEE output.

Principles
----------
* The simulation/model is not modified.
* No composite OEE score or new pass/fail gate is introduced.
* Run is the primary inferential unit wherever repeated evolutionary runs are available.
* Component-level knockout assays are reported as nested effect estimates; they are not
  treated as 252 independent evolutionary replicates.
* Channon and MODES outputs are retained as published/reference measurement families.
* Physical scaling uses paired replicate identities across ordered capacities and the
  Page trend test, which is designed for ordered alternatives in blocked/repeated data.
* The finite resolution of the exact sign-flip test is quantified explicitly.

Default input:
  ~/Downloads/REPRODUCTIVE_INTERDEPENDENCE_MULTI_HALLMARK_OEE_V12.zip

Default output:
  ~/Desktop/REPRODUCTIVE_INTERDEPENDENCE_MULTI_HALLMARK_OEE_V12_STATISTICS

Usage:
  python3 13_FORMAL_STATISTICAL_ANALYSIS_V12.py
  python3 13_FORMAL_STATISTICAL_ANALYSIS_V12.py --input /path/to/V12.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any

# Avoid nested BLAS pools during a light analysis task.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
from scipy import stats

PROGRAM_VERSION = "13.0.0-formal-statistical-analysis-v12"
ALPHA = 0.05
BOOTSTRAP_REPS = 20000
RNG_SEED = 20260823


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _auto(v: str) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    sl = s.lower()
    if sl in {"nan", "na", "none", "null"}:
        return float("nan")
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [{k: _auto(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                clean[k] = v
            w.writerow(clean)


def resolve_input(input_path: Path) -> Tuple[Path, tempfile.TemporaryDirectory | None]:
    if input_path.is_dir():
        root = input_path
        # Allow the user to pass the parent of the actual output folder.
        if not (root / "01_RUN_STATUS.csv").exists():
            candidates = [p for p in root.iterdir() if p.is_dir() and (p / "01_RUN_STATUS.csv").exists()]
            if len(candidates) == 1:
                root = candidates[0]
        return root, None
    if not zipfile.is_zipfile(input_path):
        raise FileNotFoundError(f"Input is neither a directory nor a zip archive: {input_path}")
    td = tempfile.TemporaryDirectory(prefix="v12_stats_")
    with zipfile.ZipFile(input_path) as z:
        z.extractall(td.name)
    troot = Path(td.name)
    candidates = [troot] + [p for p in troot.iterdir() if p.is_dir()]
    for p in candidates:
        if (p / "01_RUN_STATUS.csv").exists():
            return p, td
    raise FileNotFoundError("Could not locate 01_RUN_STATUS.csv inside archive")


def wilson_ci(k: int, n: int, alpha: float = ALPHA) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def exact_signflip_p_greater(values: Sequence[float]) -> float:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return float("nan")
    obs = float(np.mean(d))
    if obs <= 0 or np.all(np.abs(d) < 1e-15):
        return 1.0
    if n > 20:
        # Deterministic Monte Carlo fallback; current V12 run-level tests use n=16.
        rng = np.random.default_rng(RNG_SEED)
        B = 1_000_000
        ge = 0
        batch = 10000
        for _ in range(B // batch):
            signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, n))
            ge += int(np.count_nonzero(np.mean(signs*d[None, :], axis=1) >= obs - 1e-15))
        return (ge + 1) / (B + 1)
    total = 1 << n
    ge = 0
    for mask in range(total):
        signs = np.ones(n, dtype=float)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0
        if float(np.mean(signs*d)) >= obs - 1e-15:
            ge += 1
    return ge / total


def bootstrap_mean_ci(values: Sequence[float], reps: int = BOOTSTRAP_REPS, seed: int = RNG_SEED) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Small n (16 runs), so this is cheap and reproducible.
    idx = rng.integers(0, x.size, size=(reps, x.size))
    vals = x[idx].mean(axis=1)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.ones(m, dtype=float)
    running = 0.0
    for rank0, idx in enumerate(order):
        val = min(1.0, float(p[idx]) * (m - rank0))
        running = max(running, val)
        out[idx] = running
    return out.tolist()


def bh_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.ones(m, dtype=float)
    prev = 1.0
    for rank0 in range(m-1, -1, -1):
        idx = int(order[rank0])
        rank = rank0 + 1
        val = min(prev, float(p[idx]) * m / rank)
        q[idx] = min(1.0, val)
        prev = q[idx]
    return q.tolist()


def num(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    v = row.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def integer(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    v = row.get(key, default)
    try:
        return int(v)
    except Exception:
        return default


def causal_effect_table(assays: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in assays:
        intact = np.asarray(json.loads(str(r.get("intact_descendants_json", "[]"))), dtype=float)
        ko = np.asarray(json.loads(str(r.get("knockout_descendants_json", "[]"))), dtype=float)
        if intact.size and ko.size and intact.size == ko.size:
            logdiff = np.log1p(intact) - np.log1p(ko)
            diff = intact - ko
            sd = float(np.std(logdiff, ddof=1)) if logdiff.size > 1 else float("nan")
            dz = float(np.mean(logdiff) / sd) if np.isfinite(sd) and sd > 0 else float("nan")
            med_diff = float(np.median(diff))
            positive_pairs = int(np.count_nonzero(logdiff > 0))
            negative_pairs = int(np.count_nonzero(logdiff < 0))
            tied_pairs = int(np.count_nonzero(np.abs(logdiff) <= 1e-15))
        else:
            dz = med_diff = float("nan")
            positive_pairs = negative_pairs = tied_pairs = 0
        out.append({
            "run_id": r.get("run_id"), "component": r.get("component"),
            "generation": integer(r, "generation"), "n_carriers": integer(r, "n_carriers"),
            "repeats": integer(r, "repeats"),
            "mean_log_descendant_effect": num(r, "mean_log_descendant_effect"),
            "mean_descendant_difference": num(r, "mean_descendant_difference"),
            "median_descendant_difference": med_diff,
            "paired_standardized_effect_dz": dz,
            "positive_pairs": positive_pairs, "negative_pairs": negative_pairs, "tied_pairs": tied_pairs,
            "exact_signflip_p_one_sided": num(r, "p_value"),
            "BH_q_within_run": num(r, "q_value_BH_within_run"),
            "positive_causal_effect": integer(r, "positive_causal_effect"),
            "nominal_p_le_0_05": integer(r, "nominal_p_le_0_05"),
            "BH_q_le_0_05": integer(r, "BH_q_le_0_05"),
        })
    return out


def causal_run_summary(assays: Sequence[Mapping[str, Any]], hallmark: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_run: Dict[str, List[Mapping[str, Any]]] = {}
    for r in assays:
        by_run.setdefault(str(r.get("run_id")), []).append(r)
    hmap = {str(r.get("run_id")): r for r in hallmark}
    run_ids = sorted(set(by_run) | set(hmap))
    rows=[]
    for rid in run_ids:
        rr=by_run.get(rid,[]); h=hmap.get(rid,{})
        eff=[num(x,"mean_log_descendant_effect") for x in rr if np.isfinite(num(x,"mean_log_descendant_effect"))]
        pos=[x for x in rr if integer(x,"positive_causal_effect")==1]
        rows.append({
            "run_id":rid,
            "n_assays":len(rr),
            "n_positive_effect":sum(integer(x,"positive_causal_effect") for x in rr),
            "n_nominal_p05":sum(integer(x,"nominal_p_le_0_05") for x in rr),
            "n_BH_q05":sum(integer(x,"BH_q_le_0_05") for x in rr),
            "median_component_log_effect":float(np.median(eff)) if eff else float("nan"),
            "median_positive_component_log_effect":float(np.median([num(x,"mean_log_descendant_effect") for x in pos])) if pos else float("nan"),
            "late_positive_innovations":integer(h,"late_positive_causal_innovations"),
            "late_positive_novelty_per_1000_births":num(h,"late_positive_causal_novelty_per_1000_births"),
            "early_positive_adaptive_diversity":integer(h,"early_half_running_max_positive_causal_diversity"),
            "final_positive_adaptive_diversity":integer(h,"final_running_max_positive_causal_diversity"),
            "positive_adaptive_diversity_change":integer(h,"final_running_max_positive_causal_diversity")-integer(h,"early_half_running_max_positive_causal_diversity"),
            "late_positive_novelty_observed":integer(h,"ongoing_positive_causal_novelty_observed_over_late_half"),
            "positive_diversity_growth_observed":integer(h,"positive_causal_diversity_growth_observed"),
            "late_nominal_innovations":integer(h,"late_nominal_causal_innovations"),
        })
    return rows


def detection_resolution(run_summary: Sequence[Mapping[str, Any]], repeats: int, alpha: float=ALPHA) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pmin = 2.0 ** (-int(repeats))
    per_run=[]
    for r in run_summary:
        m=integer(r,"n_assays")
        k_needed = int(math.ceil(pmin*m/alpha - 1e-15)) if m>0 else 0
        per_run.append({
            "run_id":r.get("run_id"), "n_tests_within_run":m, "paired_repeats":repeats,
            "minimum_attainable_one_sided_exact_p":pmin,
            "single_perfect_component_can_survive_BH_q05":int(m>0 and pmin*m <= alpha + 1e-15),
            "minimum_number_of_pmin_components_needed_for_BH_q05":max(1,k_needed) if m>0 else 0,
        })
    design=[]
    for n in [8,9,10,12,16,24,32]:
        p=2.0**(-n)
        design.append({
            "paired_repeats":n,
            "minimum_attainable_one_sided_exact_p":p,
            "maximum_number_of_tests_for_single_pmin_component_to_survive_BH_q05":int(math.floor(alpha/p)),
        })
    return per_run,design


def run_level_hallmark_stats(run_summary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    n=len(run_summary)
    late=sum(integer(r,"late_positive_novelty_observed") for r in run_summary)
    growth=sum(integer(r,"positive_diversity_growth_observed") for r in run_summary)
    both=sum(int(integer(r,"late_positive_novelty_observed")==1 and integer(r,"positive_diversity_growth_observed")==1) for r in run_summary)
    rows=[]
    for name,k in [("late_positive_causal_novelty",late),("positive_causal_diversity_growth",growth),("both_in_same_run",both)]:
        lo,hi=wilson_ci(k,n)
        rows.append({"hallmark":name,"n_runs":n,"n_supporting_runs":k,"proportion":k/n if n else float("nan"),"wilson95_low":lo,"wilson95_high":hi})
    diffs=np.asarray([num(r,"positive_adaptive_diversity_change") for r in run_summary],dtype=float)
    lo,hi=bootstrap_mean_ci(diffs)
    rows.append({
        "hallmark":"paired_change_in_running_max_positive_causal_diversity",
        "n_runs":n,"n_supporting_runs":int(np.count_nonzero(diffs>0)),"proportion":float(np.mean(diffs>0)),
        "mean_change":float(np.mean(diffs)),"median_change":float(np.median(diffs)),
        "bootstrap95_mean_low":lo,"bootstrap95_mean_high":hi,
        "exact_signflip_p_greater":exact_signflip_p_greater(diffs),
    })
    return rows


def ecological_stats(eco: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    specs=[
        ("component_diversity","early_max_component_diversity","late_max_component_diversity"),
        ("functional_resource_diversity","early_max_functional_resource_diversity","late_max_functional_resource_diversity"),
        ("local_metabolic_coordinate_diversity","early_max_local_metabolic_coordinate_diversity","late_max_local_metabolic_coordinate_diversity"),
        ("genotype_diversity","early_max_genotype_diversity","late_max_genotype_diversity"),
    ]
    rows=[]; ps=[]
    for name,a,b in specs:
        diffs=np.asarray([num(r,b)-num(r,a) for r in eco],dtype=float)
        p=exact_signflip_p_greater(diffs); ps.append(p)
        lo,hi=bootstrap_mean_ci(diffs,seed=RNG_SEED+len(rows))
        rows.append({
            "metric":name,"n_runs":len(diffs),"early_mean":float(np.mean([num(r,a) for r in eco])),
            "late_mean":float(np.mean([num(r,b) for r in eco])),"mean_late_minus_early":float(np.mean(diffs)),
            "median_late_minus_early":float(np.median(diffs)),"n_positive":int(np.count_nonzero(diffs>0)),
            "n_zero":int(np.count_nonzero(diffs==0)),"n_negative":int(np.count_nonzero(diffs<0)),
            "bootstrap95_mean_diff_low":lo,"bootstrap95_mean_diff_high":hi,
            "exact_signflip_p_greater":p,
        })
    adj=holm_adjust(ps)
    for r,q in zip(rows,adj): r["holm_adjusted_p_across_4_ecological_metrics"]=q
    return rows


def modes_stats(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows=[]
    for metric in ["change","novelty","complexity","ecology"]:
        ss=[r for r in modes if str(r.get("metric"))==metric]
        if metric in {"change","novelty"}:
            k=sum(1 for r in ss if str(r.get("best_family"))=="positive")
            criterion="positive long-run time average"
        else:
            k=sum(1 for r in ss if str(r.get("best_family"))=="unbounded")
            criterion="unbounded family preferred"
        lo,hi=wilson_ci(k,len(ss))
        rows.append({"metric":metric,"criterion":criterion,"n_runs":len(ss),"n_supporting_runs":k,"proportion":k/len(ss) if ss else float("nan"),"wilson95_low":lo,"wilson95_high":hi})
    return rows


def scaling_stats(scale: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    # Same run index is a block across physical scales.
    specs=[
        "max_causally_adaptive_diversity",
        "cumulative_causally_adaptive_innovations",
        "max_component_diversity",
        "max_local_metabolic_coordinate_diversity",
    ]
    nmaxs=sorted({integer(r,"nmax") for r in scale})
    reps=sorted({int(re.search(r"run(\d+)",str(r.get("run_id"))).group(1)) for r in scale if re.search(r"run(\d+)",str(r.get("run_id")))})
    rows=[]
    for metric in specs:
        mat=[]; complete_reps=[]
        for rep in reps:
            row=[]; ok=True
            for n in nmaxs:
                matches=[r for r in scale if integer(r,"nmax")==n and re.search(r"run(\d+)",str(r.get("run_id"))) and int(re.search(r"run(\d+)",str(r.get("run_id"))).group(1))==rep]
                if len(matches)!=1 or integer(matches[0],"completed")!=1:
                    ok=False; break
                row.append(num(matches[0],metric))
            if ok:
                mat.append(row); complete_reps.append(rep)
        arr=np.asarray(mat,dtype=float)
        if arr.shape[0]>=2 and arr.shape[1]>=3:
            page=stats.page_trend_test(arr, ranked=False)
            page_stat=float(page.statistic); page_p=float(page.pvalue); method=str(page.method)
        else:
            page_stat=page_p=float("nan"); method="insufficient"
        xs=[]; ys=[]
        for r in scale:
            if integer(r,"completed")!=1: continue
            xs.append(math.log2(integer(r,"nmax"))); ys.append(num(r,metric))
        rho,p_s=stats.spearmanr(xs,ys) if len(xs)>=3 else (float("nan"),float("nan"))
        rec={
            "metric":metric,"ordered_scales":";".join(map(str,nmaxs)),"n_paired_replicates":arr.shape[0],
            "page_L_statistic":page_stat,"page_exact_p_ordered_increase":page_p,"page_method":method,
            "spearman_rho_log2_scale_descriptive":float(rho),"spearman_p_descriptive":float(p_s),
        }
        for j,n in enumerate(nmaxs):
            vals=arr[:,j] if arr.size else np.array([])
            rec[f"median_nmax_{n}"]=float(np.median(vals)) if vals.size else float("nan")
            rec[f"mean_nmax_{n}"]=float(np.mean(vals)) if vals.size else float("nan")
            rec[f"max_nmax_{n}"]=float(np.max(vals)) if vals.size else float("nan")
        rows.append(rec)
    return rows


def channon_reference(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def generate_report(outdir: Path, input_path: Path, status, assays, runsum, hallstats, ecostats, modestats, scalestats, channon, detection_rows, design_rows) -> None:
    n_runs=len(status)
    completed=sum(integer(r,"completed") for r in status)
    extinct=sum(integer(r,"extinct") for r in status)
    n_assay=len(assays)
    n_pos=sum(integer(r,"positive_causal_effect") for r in assays)
    n_nom=sum(integer(r,"nominal_p_le_0_05") for r in assays)
    n_bh=sum(integer(r,"BH_q_le_0_05") for r in assays)
    runs_any_pos=sum(1 for r in runsum if integer(r,"n_positive_effect")>0)
    runs_any_nom=sum(1 for r in runsum if integer(r,"n_nominal_p05")>0)
    late=next(r for r in hallstats if r.get("hallmark")=="late_positive_causal_novelty")
    growth=next(r for r in hallstats if r.get("hallmark")=="positive_causal_diversity_growth")
    both=next(r for r in hallstats if r.get("hallmark")=="both_in_same_run")
    paired=next(r for r in hallstats if r.get("hallmark")=="paired_change_in_running_max_positive_causal_diversity")
    pmin=min(num(r,"minimum_attainable_one_sided_exact_p") for r in detection_rows) if detection_rows else float("nan")
    max_tests=max(integer(r,"n_tests_within_run") for r in detection_rows) if detection_rows else 0
    min_repeat=next((integer(r,"paired_repeats") for r in design_rows if integer(r,"maximum_number_of_tests_for_single_pmin_component_to_survive_BH_q05")>=max_tests),None)

    lines=[]
    lines += ["# V12 formal statistical analysis", "", f"Analysis program: {PROGRAM_VERSION}", f"Input: {input_path}", f"Input SHA-256: {sha256_file(input_path) if input_path.is_file() else 'directory input'}", ""]
    lines += ["## Analysis policy", "", "The run is the primary inferential unit. Component-level knockout results are nested effect estimates and are not treated as independent evolutionary replicates. No composite OEE score or new pass/fail gate is constructed. Channon, MODES, causal intervention, ecological expansion and scaling are reported as separate evidence families.", ""]
    lines += ["## Run completion", "", f"Runs: {completed}/{n_runs} completed; extinctions: {extinct}/{n_runs}.", ""]
    lines += ["## Prospective causal knockout evidence", "", f"Assayed hereditary reaction components: {n_assay}.", f"Positive mean causal-effect estimate: {n_pos}/{n_assay}; observed in {runs_any_pos}/{n_runs} runs.", f"Nominal one-sided exact sign-flip p <= 0.05: {n_nom}/{n_assay}; present in {runs_any_nom}/{n_runs} runs.", f"Within-run BH q <= 0.05: {n_bh}/{n_assay}.", ""]
    lines += ["### Discrete-test resolution", "", f"Full-mode probes use 8 paired repeats, so the minimum attainable one-sided exact sign-flip p is {pmin:.8f}. The largest observed within-run assay family contains {max_tests} components. With this family size, a single component at the minimum possible p cannot necessarily survive BH q <= 0.05. The smallest repeat count whose p-value resolution permits a single all-direction-consistent component to survive BH for {max_tests} tests is {min_repeat}. This is a resolution statement, not a claim that {min_repeat} repeats guarantees statistical power.", ""]
    lines += ["## Ongoing adaptive-effect novelty", "", f"New positive-effect reactions occurred in the late half in {integer(late,'n_supporting_runs')}/{integer(late,'n_runs')} runs ({num(late,'proportion'):.3f}; Wilson 95% CI {num(late,'wilson95_low'):.3f}-{num(late,'wilson95_high'):.3f}).", f"Positive-effect adaptive diversity increased from the early-half running maximum to the final running maximum in {integer(growth,'n_supporting_runs')}/{integer(growth,'n_runs')} runs ({num(growth,'proportion'):.3f}; Wilson 95% CI {num(growth,'wilson95_low'):.3f}-{num(growth,'wilson95_high'):.3f}).", f"Both occurred in the same run in {integer(both,'n_supporting_runs')}/{integer(both,'n_runs')} runs ({num(both,'proportion'):.3f}; Wilson 95% CI {num(both,'wilson95_low'):.3f}-{num(both,'wilson95_high'):.3f}).", f"Across runs, the change in running-max positive-effect diversity was mean {num(paired,'mean_change'):.3f}, median {num(paired,'median_change'):.3f}; exact paired sign-flip p={num(paired,'exact_signflip_p_greater'):.6g}.", ""]
    lines += ["## Ecological expansion", ""]
    for r in ecostats:
        lines.append(f"- {r['metric']}: mean late-early={num(r,'mean_late_minus_early'):.3f}; exact sign-flip p={num(r,'exact_signflip_p_greater'):.6g}; Holm-adjusted p={num(r,'holm_adjusted_p_across_4_ecological_metrics'):.6g}.")
    lines += ["", "## MODES", ""]
    for r in modestats:
        lines.append(f"- {r['metric']}: {integer(r,'n_supporting_runs')}/{integer(r,'n_runs')} supporting runs ({num(r,'proportion'):.3f}; Wilson 95% CI {num(r,'wilson95_low'):.3f}-{num(r,'wilson95_high'):.3f}).")
    lines += ["", "## Physical-scale ordered trends", ""]
    for r in scalestats:
        lines.append(f"- {r['metric']}: Page ordered-trend p={num(r,'page_exact_p_ordered_increase'):.6g} across {integer(r,'n_paired_replicates')} paired replicate series; descriptive Spearman rho={num(r,'spearman_rho_log2_scale_descriptive'):.3f}.")
    lines += ["", "## Channon reference", ""]
    if channon:
        r=channon[0]
        lines += [f"Campaign Step-1 A_cum unbounded: {integer(r,'step1_A_cum_unbounded')}.", f"AN_new time-average positive: {integer(r,'AN_new_time_average_positive_over_observed_horizon')} (mean={num(r,'AN_new_time_average'):.6g}).", f"AN_cum unbounded: {integer(r,'AN_cum_unbounded')}; AN_median_cum unbounded: {integer(r,'AN_median_cum_unbounded')}; Step-3 observed-horizon pattern: {integer(r,'step3_observed_horizon_candidate_pattern')}.", ""]
    lines += ["## Interpretation boundary", "", "The statistics quantify finite-horizon evidence. Positive knockout-effect estimates, nominal p-values, BH-adjusted q-values, MODES, Channon activity and scaling trends are deliberately kept as separate evidence tiers. The analysis does not equate failure of any single tier with failure of OEE, and it does not claim mathematical indefinite evolution from a finite run.", ""]
    (outdir/"10_FORMAL_STATISTICAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path.home()/"Downloads"/"REPRODUCTIVE_INTERDEPENDENCE_MULTI_HALLMARK_OEE_V12.zip")
    ap.add_argument("--out", type=Path, default=Path.home()/"Desktop"/"REPRODUCTIVE_INTERDEPENDENCE_MULTI_HALLMARK_OEE_V12_STATISTICS")
    args=ap.parse_args()
    inp=args.input.expanduser().resolve(); out=args.out.expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(inp)
    root,tmp=resolve_input(inp)
    try:
        required=["01_RUN_STATUS.csv","04C_CHANNON_CAMPAIGN_CLASSIFICATION.csv","06_MODES_ASSESSMENT.csv","07_CAUSAL_ADAPTIVE_NOVELTY_ASSAYS.csv","09_OEE_HALLMARK_SUMMARY.csv","09B_ECOLOGICAL_EXPANSION_SUMMARY.csv","09C_HALLMARK_SCALABILITY_BY_RUN.csv"]
        missing=[f for f in required if not (root/f).exists()]
        if missing: raise FileNotFoundError(f"Missing V12 outputs: {missing}")
        if out.exists(): shutil.rmtree(out)
        out.mkdir(parents=True,exist_ok=True)
        print(f"[1/6] Reading V12 output: {root}")
        status=read_csv(root/"01_RUN_STATUS.csv")
        channon=read_csv(root/"04C_CHANNON_CAMPAIGN_CLASSIFICATION.csv")
        modes=read_csv(root/"06_MODES_ASSESSMENT.csv")
        assays=read_csv(root/"07_CAUSAL_ADAPTIVE_NOVELTY_ASSAYS.csv")
        hallmark=read_csv(root/"09_OEE_HALLMARK_SUMMARY.csv")
        eco=read_csv(root/"09B_ECOLOGICAL_EXPANSION_SUMMARY.csv")
        scale=read_csv(root/"09C_HALLMARK_SCALABILITY_BY_RUN.csv")
        print(f"[2/6] Causal assays: {len(assays)} components across {len(status)} runs")
        ce=causal_effect_table(assays); rs=causal_run_summary(assays,hallmark)
        repeats=max((integer(r,"repeats") for r in assays),default=8)
        det,design=detection_resolution(rs,repeats)
        print("[3/6] Run-level novelty/diversity and ecological early-late statistics")
        hs=run_level_hallmark_stats(rs); es=ecological_stats(eco)
        print("[4/6] MODES and paired physical-scale trend statistics")
        ms=modes_stats(modes); ss=scaling_stats(scale)
        print("[5/6] Writing reproducible tables")
        write_csv(out/"01_CAUSAL_COMPONENT_EFFECTS.csv",ce)
        write_csv(out/"02_CAUSAL_RUN_SUMMARY.csv",rs)
        write_csv(out/"03A_CAUSAL_BH_RESOLUTION_BY_RUN.csv",det)
        write_csv(out/"03B_CAUSAL_REPEAT_RESOLUTION_DESIGN.csv",design)
        write_csv(out/"04_RUN_LEVEL_HALLMARK_STATISTICS.csv",hs)
        write_csv(out/"05_ECOLOGICAL_EARLY_LATE_STATISTICS.csv",es)
        write_csv(out/"06_MODES_RUN_PROPORTIONS.csv",ms)
        write_csv(out/"07_PHYSICAL_SCALING_PAIRED_TRENDS.csv",ss)
        write_csv(out/"08_CHANNON_REFERENCE.csv",channon)
        manifest={
            "analysis_program_version":PROGRAM_VERSION,"input":str(inp),
            "input_sha256":sha256_file(inp) if inp.is_file() else None,"alpha":ALPHA,
            "bootstrap_reps":BOOTSTRAP_REPS,"primary_inferential_unit":"evolutionary run",
            "no_composite_oee_score":True,
        }
        (out/"00_ANALYSIS_CONFIG.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        generate_report(out,inp,status,assays,rs,hs,es,ms,ss,channon,det,design)
        print("[6/6] Complete")
        print(f"Output: {out}")
        return 0
    finally:
        if tmp is not None: tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
