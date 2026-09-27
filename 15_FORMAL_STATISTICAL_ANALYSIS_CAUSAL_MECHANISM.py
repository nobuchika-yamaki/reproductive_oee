#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_FORMAL_STATISTICAL_ANALYSIS_CAUSAL_MECHANISM.py

Formal matched-replicate analysis for
14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py.

Primary inferential unit: matched evolutionary replicate identity.
The 2 x 2 design is
    FULL       : interdependence ON,  ecological construction ON
    INT_MINUS  : interdependence OFF, ecological construction ON
    ECO_MINUS  : interdependence ON,  ecological construction OFF
    BOTH_MINUS : interdependence OFF, ecological construction OFF

For every outcome Y, replicate-level factorial contrasts are
    Interdependence = ((FULL + ECO_MINUS) - (INT_MINUS + BOTH_MINUS)) / 2
    Ecology         = ((FULL + INT_MINUS) - (ECO_MINUS + BOTH_MINUS)) / 2
    Interaction     = FULL - INT_MINUS - ECO_MINUS + BOTH_MINUS

No component-level causal assay is treated as an independent evolutionary replicate.
Resource-dependency assays are first aggregated within run, then summarized across runs.

Default input:
    ~/Desktop/REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_V14
Default output:
    ~/Desktop/REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_V14_STATISTICS
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
from scipy import stats

PROGRAM_VERSION = "15.1.0-prespecified-mechanism-statistics"
ALPHA = 0.05
BOOTSTRAP_REPS = 20000
RNG_SEED = 20260823
CONDITIONS = ("FULL", "INT_MINUS", "ECO_MINUS", "BOTH_MINUS")


def _auto(v: str) -> Any:
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    if s.lower() in {"nan", "na", "none", "null"}: return float("nan")
    try:
        if re.fullmatch(r"[-+]?\d+", s): return int(s)
        return float(s)
    except Exception:
        return s


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0: return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [{k: _auto(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader()
        for r in rows:
            z={}
            for k in keys:
                v=r.get(k,"")
                if isinstance(v,np.integer): v=int(v)
                elif isinstance(v,np.floating): v=float(v)
                z[k]=v
            w.writerow(z)


def num(r: Mapping[str,Any], k: str, default=float("nan")) -> float:
    try: return float(r.get(k,default))
    except Exception: return default


def integer(r: Mapping[str,Any], k: str, default=0) -> int:
    try: return int(r.get(k,default))
    except Exception: return default


def resolve_input(path: Path) -> Tuple[Path, tempfile.TemporaryDirectory | None]:
    if path.is_dir(): return path, None
    if not zipfile.is_zipfile(path): raise FileNotFoundError(path)
    td=tempfile.TemporaryDirectory(prefix="v14_mechanism_stats_")
    with zipfile.ZipFile(path) as z: z.extractall(td.name)
    root=Path(td.name)
    candidates=[root]+[p for p in root.rglob("02_MECHANISM_RUN_SUMMARY.csv")]
    for p in candidates:
        q=p.parent if p.is_file() else p
        if (q/"02_MECHANISM_RUN_SUMMARY.csv").exists(): return q,td
    raise FileNotFoundError("02_MECHANISM_RUN_SUMMARY.csv not found in archive")


def bootstrap_mean_ci(values: Sequence[float], seed: int=RNG_SEED) -> Tuple[float,float]:
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if x.size==0: return float("nan"),float("nan")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,x.size,size=(BOOTSTRAP_REPS,x.size))
    m=x[idx].mean(axis=1)
    return float(np.quantile(m,0.025)),float(np.quantile(m,0.975))


def exact_signflip(values: Sequence[float], alternative: str="greater") -> float:
    d=np.asarray(values,dtype=float); d=d[np.isfinite(d)]
    n=d.size
    if n==0: return float("nan")
    obs=float(np.mean(d))
    if np.all(np.abs(d)<1e-15): return 1.0
    if n>20:
        rng=np.random.default_rng(RNG_SEED); B=1_000_000; ge=0; batch=10000
        target=abs(obs) if alternative=="two-sided" else obs
        for _ in range(B//batch):
            signs=rng.choice(np.array([-1.0,1.0]),size=(batch,n)); vals=np.mean(signs*d[None,:],axis=1)
            ge += int(np.count_nonzero(np.abs(vals)>=target-1e-15)) if alternative=="two-sided" else int(np.count_nonzero(vals>=target-1e-15))
        return (ge+1)/(B+1)
    total=1<<n; ge=0; target=abs(obs) if alternative=="two-sided" else obs
    for mask in range(total):
        signs=np.ones(n)
        for i in range(n):
            if (mask>>i)&1: signs[i]=-1.0
        val=float(np.mean(signs*d))
        if (abs(val)>=target-1e-15) if alternative=="two-sided" else (val>=target-1e-15): ge+=1
    return ge/total


def summarize_values(name: str, values: Sequence[float], seed: int) -> Dict[str,Any]:
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    lo,hi=bootstrap_mean_ci(x,seed)
    return {"contrast":name,"n_matched_replicates":int(x.size),"mean":float(np.mean(x)) if x.size else float("nan"),
            "median":float(np.median(x)) if x.size else float("nan"),"sd":float(np.std(x,ddof=1)) if x.size>1 else float("nan"),
            "bootstrap95_mean_low":lo,"bootstrap95_mean_high":hi,
            "exact_signflip_p_greater":exact_signflip(x,"greater"),"exact_signflip_p_two_sided":exact_signflip(x,"two-sided"),
            "n_positive":int(np.count_nonzero(x>0)),"n_zero":int(np.count_nonzero(np.abs(x)<=1e-15)),"n_negative":int(np.count_nonzero(x<0))}




def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p=np.asarray(list(pvals),dtype=float)
    out=np.full(p.size,np.nan,dtype=float)
    finite=np.flatnonzero(np.isfinite(p))
    if finite.size==0:
        return out.tolist()
    order=finite[np.argsort(p[finite])]
    running=0.0
    m=order.size
    for rank0,idx in enumerate(order):
        val=min(1.0,float(p[idx])*(m-rank0))
        running=max(running,val)
        out[idx]=running
    return out.tolist()


def add_holm(rows: List[Dict[str,Any]], selector, field: str="exact_signflip_p_greater", outfield: str="holm_adjusted_p") -> None:
    idx=[i for i,r in enumerate(rows) if selector(r)]
    adj=holm_adjust([num(rows[i],field) for i in idx])
    for i,q in zip(idx,adj):
        rows[i][outfield]=q


def matched_map(rows: Sequence[Mapping[str,Any]], outcome: str) -> Dict[int,Dict[str,float]]:
    out: Dict[int,Dict[str,float]]=defaultdict(dict)
    for r in rows:
        rep=integer(r,"replicate",-1); c=str(r.get("condition",""))
        v=num(r,outcome)
        if rep>=0 and c in CONDITIONS and np.isfinite(v): out[rep][c]=v
    return out


def factorial_stats(rows: Sequence[Mapping[str,Any]], outcome: str, seed_offset: int=0) -> Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    mm=matched_map(rows,outcome); raw=[]
    I=[]; E=[]; X=[]
    for rep in sorted(mm):
        d=mm[rep]
        if not all(c in d for c in CONDITIONS): continue
        f,i,e,b=(d["FULL"],d["INT_MINUS"],d["ECO_MINUS"],d["BOTH_MINUS"])
        ci=((f+e)-(i+b))/2.0; ce=((f+i)-(e+b))/2.0; cx=f-i-e+b
        I.append(ci); E.append(ce); X.append(cx)
        raw.append({"outcome":outcome,"replicate":rep,"FULL":f,"INT_MINUS":i,"ECO_MINUS":e,"BOTH_MINUS":b,
                    "interdependence_main_effect":ci,"ecology_main_effect":ce,"interaction":cx})
    stats_rows=[]
    for j,(name,vals) in enumerate((("interdependence_main_effect",I),("ecology_main_effect",E),("interaction",X))):
        z=summarize_values(name,vals,RNG_SEED+seed_offset+j); z["outcome"]=outcome; stats_rows.append(z)
    return raw,stats_rows


def condition_descriptives(rows: Sequence[Mapping[str,Any]], outcomes: Sequence[str]) -> List[Dict[str,Any]]:
    out=[]
    for outcome in outcomes:
        for c in CONDITIONS:
            x=np.asarray([num(r,outcome) for r in rows if str(r.get("condition"))==c],dtype=float); x=x[np.isfinite(x)]
            out.append({"outcome":outcome,"condition":c,"n":int(x.size),"mean":float(np.mean(x)) if x.size else float("nan"),
                        "sd":float(np.std(x,ddof=1)) if x.size>1 else float("nan"),"median":float(np.median(x)) if x.size else float("nan"),
                        "min":float(np.min(x)) if x.size else float("nan"),"max":float(np.max(x)) if x.size else float("nan")})
    return out


def resource_run_stats(rows: Sequence[Mapping[str,Any]], allruns: Sequence[Mapping[str,Any]]) -> Tuple[List[Dict[str,Any]],Dict[str,Any]]:
    by: Dict[Tuple[int,str],List[Mapping[str,Any]]]=defaultdict(list)
    for r in rows: by[(integer(r,"replicate",-1),str(r.get("condition","")))].append(r)
    out=[]
    keys={(integer(r,"replicate",-1),str(r.get("condition",""))) for r in allruns if integer(r,"replicate",-1)>=0 and str(r.get("condition","")) in CONDITIONS}
    keys.update(by.keys())
    for (rep,c) in sorted(keys):
        rr=by.get((rep,c),[])
        strict=[r for r in rr if integer(r,"strict_shared_dependency_candidate")==1]
        effects=[num(r,"mean_log_erasure_effect") for r in strict if np.isfinite(num(r,"mean_log_erasure_effect"))]
        resc=[num(r,"rescue_matches_intact_fraction") for r in strict if np.isfinite(num(r,"rescue_matches_intact_fraction"))]
        out.append({"replicate":rep,"condition":c,"n_resource_assays":len(rr),"n_strict_candidates":len(strict),
                    "n_positive_strict":sum(num(r,"mean_log_erasure_effect")>0 for r in strict),
                    "mean_strict_log_erasure_effect":float(np.mean(effects)) if effects else float("nan"),
                    "median_strict_log_erasure_effect":float(np.median(effects)) if effects else float("nan"),
                    "mean_rescue_matches_intact_fraction":float(np.mean(resc)) if resc else float("nan")})
    full=[r for r in out if r["condition"]=="FULL"]
    effects=[num(r,"mean_strict_log_erasure_effect") for r in full if integer(r,"n_strict_candidates")>0 and np.isfinite(num(r,"mean_strict_log_erasure_effect"))]
    summary={"FULL_runs":len(full),"FULL_runs_with_strict_candidate":sum(integer(r,"n_strict_candidates")>0 for r in full),
             "FULL_runs_with_positive_strict_dependency":sum(integer(r,"n_positive_strict")>0 for r in full),
             "mean_run_level_erasure_effect":float(np.mean(effects)) if effects else float("nan"),
             "exact_signflip_p_greater_conditional_on_candidate":exact_signflip(effects,"greater") if effects else float("nan")}
    return out,summary


def dependency_growth(rows: Sequence[Mapping[str,Any]], status: Sequence[Mapping[str,Any]]) -> Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    requested={str(r.get("run_id")):integer(r,"generations_requested") for r in status}
    by: Dict[str,List[Mapping[str,Any]]]=defaultdict(list)
    for r in rows:
        if str(r.get("condition"))=="FULL": by[str(r.get("run_id"))].append(r)
    metrics=("n_dependency_edges","n_cross_genotype_edges","n_historically_ordered_edges","dependency_depth")
    runrows=[]
    for rid,rr in by.items():
        half=requested.get(rid,0)/2.0; early=[r for r in rr if num(r,"generation")<half]; late=[r for r in rr if num(r,"generation")>=half]
        rec={"run_id":rid,"replicate":integer(rr[0],"replicate",-1)}
        for m in metrics:
            a=max((num(r,m,0.0) for r in early),default=0.0); b=max((num(r,m,0.0) for r in late),default=0.0)
            rec[f"early_max_{m}"]=a; rec[f"late_max_{m}"]=b; rec[f"late_minus_early_{m}"]=b-a
        runrows.append(rec)
    statrows=[]
    for j,m in enumerate(metrics):
        vals=[num(r,f"late_minus_early_{m}") for r in runrows]
        z=summarize_values(f"late_minus_early_{m}",vals,RNG_SEED+100+j); z["metric"]=m; statrows.append(z)
    return runrows,statrows


def main() -> int:
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input",type=Path,default=Path.home()/"Desktop"/"REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_V14")
    ap.add_argument("--out",type=Path,default=Path.home()/"Desktop"/"REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_V14_STATISTICS")
    args=ap.parse_args(); root,tmp=resolve_input(args.input.expanduser().resolve()); out=args.out.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    try:
        run=read_csv(root/"02_MECHANISM_RUN_SUMMARY.csv"); eq=read_csv(root/"08_EQUAL_BIRTH_OPPORTUNITY.csv")
        resources=read_csv(root/"04_RESOURCE_DEPENDENCY_ASSAYS.csv"); deps=read_csv(root/"05_DEPENDENCY_NETWORK_TIMESERIES.csv"); status=read_csv(root/"01_RUN_STATUS.csv")
        primary_outcomes=(
            "late_positive_causal_novelty_per_1000_births",
            "final_running_max_positive_causal_diversity",
        )
        secondary_outcomes=(
            "cumulative_positive_causal_innovations",
            "max_dependency_edges",
            "max_cross_genotype_dependency_edges",
            "max_historically_ordered_dependency_edges",
            "max_dependency_depth",
        )
        diagnostic_outcomes=("analysis_births","structural_mutation_events","reaction_identities_ever")
        raw=[]; fst=[]
        all_factorial_outcomes=primary_outcomes+secondary_outcomes+diagnostic_outcomes
        for j,o in enumerate(all_factorial_outcomes):
            a,b=factorial_stats(run,o,10*j)
            role=("primary" if o in primary_outcomes else "secondary" if o in secondary_outcomes else "diagnostic")
            for r in a: r["analysis_role"]=role
            for r in b: r["analysis_role"]=role
            raw.extend(a); fst.extend(b)

        # Prespecified multiplicity families from Methods 2.14.
        # Primary family: the interaction term only, across the two primary outcomes.
        add_holm(
            fst,
            lambda r: r.get("analysis_role")=="primary" and r.get("contrast")=="interaction",
            outfield="holm_adjusted_p_primary_interaction_family",
        )
        # Secondary mechanistic family: all three factorial contrasts across five secondary outcomes.
        add_holm(
            fst,
            lambda r: r.get("analysis_role")=="secondary",
            outfield="holm_adjusted_p_secondary_mechanistic_family",
        )
        eq_outcomes=("positive_causal_diversity","cumulative_positive_causal_innovations","component_diversity","structural_mutation_events_cumulative","reaction_identities_ever")
        eqraw=[]; eqstats=[]
        for j,o in enumerate(eq_outcomes):
            a,b=factorial_stats(eq,o,500+10*j); eqraw.extend(a); eqstats.extend(b)
        desc=condition_descriptives(run,primary_outcomes+secondary_outcomes+diagnostic_outcomes)
        resource_runs,resource_summary=resource_run_stats(resources,run)
        dep_runs,dep_stats=dependency_growth(deps,status)
        add_holm(
            dep_stats,
            lambda r: True,
            outfield="holm_adjusted_p_dependency_growth_family",
        )
        write_csv(out/"01_CONDITION_DESCRIPTIVES.csv",desc); write_csv(out/"02_FACTORIAL_CONTRASTS_BY_REPLICATE.csv",raw); write_csv(out/"03_FACTORIAL_INFERENCE.csv",fst)
        write_csv(out/"04_EQUAL_BIRTH_FACTORIAL_CONTRASTS_BY_REPLICATE.csv",eqraw); write_csv(out/"05_EQUAL_BIRTH_FACTORIAL_INFERENCE.csv",eqstats)
        write_csv(out/"06_RESOURCE_DEPENDENCY_BY_RUN.csv",resource_runs); write_csv(out/"07_FULL_DEPENDENCY_GROWTH_BY_RUN.csv",dep_runs); write_csv(out/"08_FULL_DEPENDENCY_GROWTH_INFERENCE.csv",dep_stats)
        report=["# Formal statistical analysis: causal mechanism validation","",f"Program version: {PROGRAM_VERSION}","",
                "## Design","","The evolutionary run is the inferential unit. The four mechanism conditions are matched by replicate seed. Factorial main effects and the interaction are computed within each complete four-condition replicate before inference across replicates.","",
                "## Prespecified primary outcomes","",*['- '+x for x in primary_outcomes],"",
                "Primary inference is the interaction contrast across these two outcomes; the two one-sided interaction p values are Holm-adjusted as one family.","",
                "## Secondary mechanistic outcomes","",*['- '+x for x in secondary_outcomes],"",
                "All one-sided factorial contrast p values across the five secondary outcomes are Holm-adjusted within one secondary mechanistic family.","",
                "## Evolutionary-opportunity diagnostics","",*['- '+x for x in diagnostic_outcomes],"",
                "## Resource dependency","",json.dumps(resource_summary,indent=2),"",
                "## Dependency-network growth","","The four FULL-condition early-versus-late dependency-growth tests are Holm-adjusted as one family.","",
                "## Interpretation boundary","","Component-level knockout and resource-erasure assays are nested mechanistic observations. They are not counted as independent evolutionary replicates. Exact sign-flip tests and bootstrap confidence intervals are applied to replicate-level contrasts. No composite score is constructed.",""]
        (out/"09_STATISTICAL_REPORT.md").write_text("\n".join(report),encoding="utf-8")
        print(json.dumps({"program_version":PROGRAM_VERSION,"n_run_rows":len(run),"n_factorial_rows":len(fst),"primary_outcomes":list(primary_outcomes),"secondary_outcomes":list(secondary_outcomes),"resource_summary":resource_summary,"output":str(out)},indent=2))
        return 0
    finally:
        if tmp is not None: tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
