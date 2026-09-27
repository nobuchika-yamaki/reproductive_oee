#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py

Benchmark-native open-ended extension of the reproductive-interdependence model.

Scientific design
-----------------
The inherited core mechanism is retained:
    transient functional unreliability
      -> facultative pair interaction
      -> functional sharing / buffering
      -> tolerated loss of individual basal functions
      -> complementary specialization and reproductive interdependence.

The fixed K=8 evolvable functional repertoire of the original model is NOT used as
an evolutionary ceiling.  Instead, every cell also carries a variable-length ordered
list of hereditary reaction transformations (input substrate -> output substrate).
Reaction endpoints are non-negative integers and mutation is a local random walk on
those integers.  No stage sequence, target innovation, sex class, mating type,
predefined niche list, recursive innovation ladder, or OEE score is encoded.
Constructed products persist transiently in a dissipative shared environment and can
become substrates for later reaction transformations.  V11 removes the finite
8-function hash previously used to assign reaction products a phenotype.  Every
substrate token is now its own metabolic coordinate.  A fixed fraction of a produced
constructed token is retained locally and can be consumed immediately only by an
exact downstream reaction in the same hereditary reaction network.  Residual local
material is then released to the shared environment.  Thus R(A->B) and R(A->C) have
different consequences only when the exact B or C coordinate is actually usable by
the inherited network or the external ecology.  No token receives a reward for being
new, large, rare, late, or benchmark-significant.

OEE evaluation
--------------
The model dynamics are evaluated through multiple published/established OEE hallmarks.
No single benchmark is treated as the definition of OEE and no composite score is used.

PRIMARY BEHAVIORAL-HALLMARK LAYER
* Ongoing adaptive novelty: a hereditary reaction transformation that has persisted
  across consecutive observation snapshots is tested by a paired prospective knockout.
  The intact and knockout state are cloned from the same live population/environment;
  structural and quantitative mutation are disabled during the one-generation probe so
  the intervention isolates the current causal reproductive contribution. Tagged
  descendants of current carriers are compared across paired repeats. The causal effect
  direction, one-sided exact sign-flip p-value, and within-run Benjamini-Hochberg q-value
  are all retained as separate evidence tiers rather than collapsed into a single gate.
* Ongoing novelty is the time-resolved accumulation/rate of newly causally validated
  hereditary reaction transformations.
* Adaptive complexity/diversity is the number and running upper bound of simultaneously
  present causally validated hereditary reaction transformations.
* Ecological expansion is reported from constructed-resource, metabolic-coordinate,
  genotype and MODES ecology trajectories; it is not collapsed into a score.

INDEPENDENT REFERENCE LAYERS
* MODES (Dolson et al. 2019): change, novelty, complexity and ecology.
* Channon (2024): Steps 1-3 activity statistics are retained as an independent reference
  benchmark. Channon threshold crossing is NOT allowed to override direct causal evidence.
* Physical-scale runs can be requested independently of Channon Step 3; scale-dependent
  causal-adaptive diversity, novelty and ecological expansion are reported directly.

There is deliberately NO causal-depth gate, niche-depth gate, ratchet score, bespoke OEE
score, majority-vote pass rule, or requirement that all benchmark families agree.

Efficiency safeguards
---------------------
* Process-level parallelism across independent runs.
* BLAS/OpenMP thread pools forced to one thread per worker.
* Population state stored in NumPy arrays.
* Variable-length reaction genomes are interned: cells carry int32 genotype IDs,
  and identical genomes share one immutable representation.
* Genotype mutation is event-driven: only offspring with a structural mutation
  enter Python-level mutation code; unchanged offspring inherit genotype IDs directly.
* Environmental reaction fluxes are evaluated by active genotype counts and NumPy
  bincount operations rather than cell x module nested loops.
* MODES lineage persistence is evaluated online with a fixed-depth ancestry ring;
  the complete genealogy is never retained.
* Neutral-shadow states are reset at snapshots and only their current genomes are
  stored; no shadow genealogy or environment is simulated.
* Trajectory I/O is snapshot-only and gzip-compressed per run.
* Safety limits abort rather than truncate scientific state, so they do not create
  an artificial evolutionary ceiling.

Recommended commands
--------------------
Self-test:
    python3 14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py --self-test

Engineering smoke:
    python3 14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py --mode smoke --through 3 --workers 1

Full Steps 1-3 + MODES:
    python3 14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py --mode full --through 3 --workers auto

Full Steps 1-5:
    python3 14_REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_VALIDATION.py --mode full --through 5 --workers auto

Default output:
    ~/Desktop/REPRODUCTIVE_INTERDEPENDENCE_MULTI_HALLMARK_OEE_V12
"""

from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import re
import multiprocessing as mp
import statistics
import sys
import time
import traceback
import tempfile
import shutil
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import curve_fit

PROGRAM_VERSION = "14.0.0-causal-mechanism-factorial"
OUTPUT_PREFIX = "14"
BASE_CAPACITY = 192
REFERENCES = {
    "channon_2024": {
        "citation": "Channon A. A Procedure for Testing for Tokyo Type 1 Open-Ended Evolution. Artificial Life. 2024;30(3):345-355.",
        "doi": "10.1162/artl_a_00430",
    },
    "dolson_2019": {
        "citation": "Dolson EL, Vostinar AE, Wiser MJ, Ofria C. The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems. Artificial Life. 2019;25(1):50-73.",
        "doi": "10.1162/artl_a_00280",
    },
    "channon_2019": {
        "citation": "Channon A. Maximum Individual Complexity is Indefinitely Scalable in Geb. Artificial Life. 2019;25(2):134-144.",
        "doi": "10.1162/artl_a_00285",
    },
}


# =============================================================================
# Generic deterministic I/O utilities
# =============================================================================

def stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b("|".join(str(x) for x in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Path):
        return str(v)
    return v


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(json_safe(obj), indent=2, ensure_ascii=False))


def write_json_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        atomic_text(path, "")
        return
    fields: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json_safe(r.get(k)) for k in fields})
    os.replace(tmp, path)


def auto_workers(requested: str, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if str(requested).lower() != "auto":
        return max(1, min(int(requested), n_tasks))
    cpus = os.cpu_count() or 2
    # Empirically, four independent NumPy/SciPy workers give substantially better
    # throughput than eight on memory-constrained desktop systems; above that,
    # process and memory-bandwidth contention can dominate.  Users may still
    # explicitly request a larger worker count with --workers N.
    return max(1, min(n_tasks, max(1, cpus - 1), 4))


# =============================================================================
# Model state
# =============================================================================

Module = Tuple[int, int]
GenomeKey = Tuple[int, Tuple[Module, ...]]  # fixed basal marker, ordered open-ended reaction modules


@dataclass(frozen=True)
class ModelConfig:
    # Population / physical capacity
    n0: int = BASE_CAPACITY
    nmax: int = BASE_CAPACITY
    basal_functions: int = 8

    # Mechanism-validation switches. BASE/FULL preserve the V12 scientific model.
    # INT_MINUS retains pair formation but removes partner functional rescue and
    # regulatory sharing. ECO_MINUS retains hereditary reactions and local cascades
    # but prevents reaction products from entering the shared environment.
    mechanism_condition: str = "BASE"
    functional_interdependence_enabled: bool = True
    shared_ecological_construction_enabled: bool = True

    # Reproduction and interaction -- inherited from reproductive-interdependence logic
    r0: float = 2.0
    transient_failure_probability: float = 0.020
    p0_fusion: float = 0.05
    passive_persistence: float = 0.05
    fusion_event_cost: float = 0.10
    persistent_pair_cost: float = 0.0525
    basal_expression_cost: float = 0.35
    theta: float = 1.0
    initial_x: float = 1.20
    x_min: float = 0.0
    x_max: float = 2.0
    mu_x: float = 1.0e-3
    sigma_x: float = 0.02
    mu_loss: float = 1.0e-4
    loss_factor_min: float = 0.0
    loss_factor_max: float = 0.5
    regulatory_sharing_efficiency: float = 0.25
    reaction_expression_cost: float = 0.012
    fusogenicity_cost: float = 0.05
    adhesion_cost: float = 0.010
    # Shared dissipative reaction environment
    source_per_capacity: float = 0.55
    substrate_dissipation: float = 0.070
    reaction_rate: float = 0.32
    half_saturation: float = 18.0
    harvest_fraction: float = 0.18
    # The V5 reproductive core remains viable without the open-ended reaction layer.
    # Hereditary reactions provide an additional, bounded energetic benefit; they are
    # innovations, not an imposed single point of metabolic failure.
    reaction_energy_bonus: float = 0.30
    metabolic_half_energy: float = 0.020

    # Exact-token metabolic coordinates.  Constructed token identities are not
    # projected onto the fixed K=8 basal-function vector.  A retained product can
    # instead feed an exact downstream reaction in the same hereditary network.
    # This gives every token an independent functional coordinate without any
    # novelty reward or predefined innovation ladder.
    local_product_retention_fraction: float = 0.20

    initial_substrate_mass_per_capacity: float = 1.5

    # Structural reaction mutation. Basal functions evolve quantitatively in
    # PopulationState.x and are not structural Channon components.
    mu_module_point_per_module: float = 1.0e-4
    mu_module_dup_per_module: float = 5.0e-5
    mu_module_del_per_module: float = 2.0e-5
    mu_module_add: float = 1.0e-4
    endpoint_geometric_p: float = 0.62
    mu_phi: float = 0.008
    sigma_phi: float = 0.018
    mu_alpha: float = 0.008
    sigma_alpha: float = 0.018

    # Observational schedule
    generations: int = 40_000
    snapshot_every: int = 100
    modes_filter_generations: int = 32
    validation_every: int = 1_000

    # Direct causal-adaptation observation.  A component is probed once after it has
    # survived across two consecutive snapshots.  Full mode uses 8 paired repeats;
    # smoke mode shortens this only for engineering validation.
    causal_probe_repeats: int = 8
    causal_probe_alpha: float = 0.05

    # Safety aborts only.  They NEVER truncate genomes or token state.
    safety_max_modules_per_genome: int = 100_000
    safety_max_token_id: int = 2_000_000
    safety_max_genotypes: int = 2_000_000
    safety_max_population: int = 200_000


@dataclass(frozen=True)
class RunTask:
    run_id: str
    seed: int
    cfg: ModelConfig
    physical_scale: float = 1.0
    scaling: bool = False


@dataclass
class PopulationState:
    gids: np.ndarray          # int32 reaction-genotype IDs
    x: np.ndarray             # float32 [n, basal_functions], quantitative inherited functions
    phi: np.ndarray           # float32
    alpha: np.ndarray         # float32
    partner: np.ndarray       # int32 reciprocal link, -1 free
    individual_id: np.ndarray # int64
    ancestors: np.ndarray     # int64 [n, filter_depth], parent at col0


class GenotypePool:
    """Intern immutable variable-length genomes and cache module arrays."""

    def __init__(self, basal_functions: int):
        self.basal_functions = int(basal_functions)
        self.keys: List[GenomeKey] = []
        self.lookup: Dict[GenomeKey, int] = {}
        self.inputs: List[np.ndarray] = []
        self.outputs: List[np.ndarray] = []
        self.module_counts: List[int] = []
        self.basal_counts: List[int] = []
        # Dense numeric caches make per-generation state lookup O(n) NumPy indexing
        # rather than repeated Python fromiter loops. Arrays grow geometrically.
        self._cache_capacity = 256
        self.basal_mask_arr = np.zeros(self._cache_capacity, dtype=np.uint64)
        self.basal_count_arr = np.zeros(self._cache_capacity, dtype=np.int16)
        self.module_count_arr = np.zeros(self._cache_capacity, dtype=np.int32)
        self.max_token: int = 1
        # Evolutionary-opportunity accounting for the live evolutionary pool.
        self.structural_mutation_events: int = 0
        self.component_identities_ever: Set[str] = set()

    def _ensure_cache(self, gid: int) -> None:
        if gid < self._cache_capacity:
            return
        newcap = self._cache_capacity
        while newcap <= gid:
            newcap *= 2
        bm = np.zeros(newcap, dtype=np.uint64); bm[:self._cache_capacity] = self.basal_mask_arr
        bc = np.zeros(newcap, dtype=np.int16); bc[:self._cache_capacity] = self.basal_count_arr
        mc = np.zeros(newcap, dtype=np.int32); mc[:self._cache_capacity] = self.module_count_arr
        self.basal_mask_arr, self.basal_count_arr, self.module_count_arr = bm, bc, mc
        self._cache_capacity = newcap

    def intern(self, key: GenomeKey) -> int:
        mask, modules = key
        mask = int(mask)
        modules = tuple((int(a), int(b)) for a, b in modules)
        key = (mask, modules)
        old = self.lookup.get(key)
        if old is not None:
            return old
        gid = len(self.keys)
        self.lookup[key] = gid
        self.keys.append(key)
        if modules:
            ia = np.fromiter((m[0] for m in modules), dtype=np.int32, count=len(modules))
            oa = np.fromiter((m[1] for m in modules), dtype=np.int32, count=len(modules))
            self.max_token = max(self.max_token, int(ia.max(initial=0)), int(oa.max(initial=0)))
        else:
            ia = np.empty(0, dtype=np.int32)
            oa = np.empty(0, dtype=np.int32)
        self.inputs.append(ia)
        self.outputs.append(oa)
        mcount = len(modules); bcount = int(mask.bit_count())
        self.module_counts.append(mcount)
        self.basal_counts.append(bcount)
        self._ensure_cache(gid)
        self.basal_mask_arr[gid] = np.uint64(mask)
        self.basal_count_arr[gid] = bcount
        self.module_count_arr[gid] = mcount
        for a, b in modules:
            self.component_identities_ever.add(f"R{int(a)}>{int(b)}")
        return gid

    def key(self, gid: int) -> GenomeKey:
        return self.keys[int(gid)]

    def components(self, gid: int) -> Tuple[str, ...]:
        # Channon component mapping follows the prior standard benchmark:
        # only discrete hereditary reaction transformations are counted.
        # Quantitative basal investment is phenotype/fitness state, not a new component.
        _, modules = self.keys[int(gid)]
        return tuple(f"R{a}>{b}" for a, b in modules)

    def module_len(self, gid: int) -> int:
        return self.module_counts[int(gid)]

    def total_function_len(self, gid: int) -> int:
        return self.module_counts[int(gid)]


# =============================================================================
# Local structural mutation -- no stage ladder and no finite function alphabet
# =============================================================================

def _endpoint_step(rng: np.random.Generator, p: float) -> int:
    # Geometric support 1,2,...; random sign; reflecting boundary at zero.
    mag = int(rng.geometric(p))
    return mag if rng.random() < 0.5 else -mag


def _mutate_endpoint(v: int, rng: np.random.Generator, p: float) -> int:
    return max(0, int(v) + _endpoint_step(rng, p))


def mutation_rate(key: GenomeKey, cfg: ModelConfig) -> float:
    _, modules = key
    l = len(modules)
    return (
        l * (cfg.mu_module_point_per_module + cfg.mu_module_dup_per_module + cfg.mu_module_del_per_module)
        + cfg.mu_module_add
    )


def mutate_genome_key(key: GenomeKey, rng: np.random.Generator, cfg: ModelConfig) -> GenomeKey:
    """Apply exactly one structural mutation event, conditional on an event occurring."""
    mask, modules_t = key
    modules = list(modules_t)
    l = len(modules)
    rates = [
        ("point", l * cfg.mu_module_point_per_module),
        ("dup", l * cfg.mu_module_dup_per_module),
        ("del", l * cfg.mu_module_del_per_module),
        ("add", cfg.mu_module_add),
    ]
    total = sum(x[1] for x in rates)
    if total <= 0.0:
        return key
    z = rng.random() * total
    acc = 0.0
    event = rates[-1][0]
    for name, r in rates:
        acc += r
        if z <= acc:
            event = name
            break

    if event == "point" and modules:
        j = int(rng.integers(len(modules)))
        a, b = modules[j]
        if rng.random() < 0.5:
            a = _mutate_endpoint(a, rng, cfg.endpoint_geometric_p)
        else:
            b = _mutate_endpoint(b, rng, cfg.endpoint_geometric_p)
        modules[j] = (a, b)
    elif event == "dup" and modules:
        j = int(rng.integers(len(modules)))
        a, b = modules[j]
        # Gene duplication is local.  Most copies are exact; a minority are born
        # neofunctionalized by the same endpoint random-walk operator.
        if rng.random() < 0.45:
            if rng.random() < 0.5:
                a = _mutate_endpoint(a, rng, cfg.endpoint_geometric_p)
            else:
                b = _mutate_endpoint(b, rng, cfg.endpoint_geometric_p)
        pos = int(rng.integers(len(modules) + 1))
        modules.insert(pos, (a, b))
    elif event == "del" and modules:
        del modules[int(rng.integers(len(modules)))]
    elif event == "add":
        if modules:
            a, b = modules[int(rng.integers(len(modules)))]
            # A new module is a locally mutated copy; there is no rule forcing its
            # input to equal a previous output or forcing a sequence of stages.
            if rng.random() < 0.5:
                a = _mutate_endpoint(a, rng, cfg.endpoint_geometric_p)
            if rng.random() < 0.5 or a == modules[0][0]:
                b = _mutate_endpoint(b, rng, cfg.endpoint_geometric_p)
            pos = int(rng.integers(len(modules) + 1))
            modules.insert(pos, (a, b))
        else:
            modules.append((0, 1))

    return int(mask), tuple(modules)


def mutate_offspring_gids(
    parent_gids: np.ndarray,
    pool: GenotypePool,
    rng: np.random.Generator,
    cfg: ModelConfig,
) -> np.ndarray:
    out = np.asarray(parent_gids, dtype=np.int32).copy()
    if out.size == 0:
        return out
    # Group by parent genotype so the probability vector is computed once per gid.
    unique, inv = np.unique(out, return_inverse=True)
    per_module = cfg.mu_module_point_per_module + cfg.mu_module_dup_per_module + cfg.mu_module_del_per_module
    rates = pool.module_count_arr[unique].astype(np.float64, copy=False) * per_module + cfg.mu_module_add
    p = 1.0 - np.exp(-rates[inv])
    mut_idx = np.flatnonzero(rng.random(out.size) < p)
    pool.structural_mutation_events += int(mut_idx.size)
    for i in mut_idx:
        pgid = int(out[i])
        new_key = mutate_genome_key(pool.key(pgid), rng, cfg)
        if len(new_key[1]) > cfg.safety_max_modules_per_genome:
            raise RuntimeError("SAFETY_ABORT modules_per_genome exceeded")
        if new_key[1]:
            mt = max(max(a, b) for a, b in new_key[1])
            if mt > cfg.safety_max_token_id:
                raise RuntimeError("SAFETY_ABORT token_id exceeded")
        out[i] = pool.intern(new_key)
        if len(pool.keys) > cfg.safety_max_genotypes:
            raise RuntimeError("SAFETY_ABORT genotype_pool exceeded")
    return out


# =============================================================================
# Shared dissipative reaction environment
# =============================================================================

class ReactionEnvironment:
    """Shared chemistry plus exact-token private metabolic routing.

    V11 treats every token as an independent metabolic coordinate.  Locally retained
    product is processed by exact matching downstream reactions in the same hereditary
    network under a fast-local-equilibration assumption.  Because every local reaction
    removes the same harvest fraction h>0, the complete downstream routing is a finite
    linear absorbing-flow problem even when the reaction graph contains cycles.  Its
    solution is cached once per immutable genotype.  This avoids iterative cascade
    simulation and keeps per-generation work proportional to active modules/genotypes.

    The solver contains no evolutionary depth cap: cycles are handled analytically by
    (I-(1-h)Q)^-1.  Token identity, novelty, lineage, generation and benchmark state
    never enter the fitness rule.
    """

    def __init__(self, cfg: ModelConfig):
        self.mass = np.zeros(8, dtype=np.float64)
        self.mass[0] = cfg.initial_substrate_mass_per_capacity * cfg.nmax
        self.total_external_input = float(self.mass[0])
        self.total_harvest = 0.0
        self.max_active_substrates = 1
        self._local_flow_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def ensure(self, token: int) -> None:
        token = int(token)
        if token < self.mass.size:
            return
        newn = max(token + 1, int(self.mass.size * 1.6) + 8)
        arr = np.zeros(newn, dtype=np.float64)
        arr[:self.mass.size] = self.mass
        self.mass = arr

    def _local_flow(self, gid: int, pool: GenotypePool, cfg: ModelConfig) -> Optional[Dict[str, Any]]:
        """Return cached exact-token absorbing flow for one immutable genotype.

        None means no constructed output of this genotype can feed any of its own
        reaction inputs, so all locally retained material is released unchanged.
        """
        gid = int(gid)
        if gid in self._local_flow_cache:
            return self._local_flow_cache[gid]
        ins = pool.inputs[gid].astype(np.int64, copy=False)
        outs = pool.outputs[gid].astype(np.int64, copy=False)
        if ins.size == 0:
            self._local_flow_cache[gid] = None
            return None
        input_set = set(int(x) for x in ins)
        if not any(int(o) > 0 and int(o) in input_set for o in outs):
            self._local_flow_cache[gid] = None
            return None
        h = float(cfg.harvest_fraction)
        if not (0.0 < h < 1.0):
            raise RuntimeError("local exact-token routing requires 0 < harvest_fraction < 1")

        tokens = np.unique(np.concatenate((ins, outs))).astype(np.int64, copy=False)
        index = {int(t): i for i, t in enumerate(tokens)}
        transient_tokens = np.array(sorted(input_set), dtype=np.int64)
        terminal_tokens = np.array([int(t) for t in tokens if int(t) not in input_set], dtype=np.int64)
        ti = {int(t): i for i, t in enumerate(transient_tokens)}
        ai = {int(t): i for i, t in enumerate(terminal_tokens)}
        nt, na = len(transient_tokens), len(terminal_tokens)
        Q = np.zeros((nt, nt), dtype=np.float64)
        R = np.zeros((nt, na), dtype=np.float64)
        for t in transient_tokens:
            src = int(t)
            dests = outs[ins == src]
            if dests.size == 0:
                continue
            w = 1.0 / float(dests.size)
            row = ti[src]
            for d in dests:
                dst = int(d)
                if dst in ti:
                    Q[row, ti[dst]] += w
                else:
                    R[row, ai[dst]] += w

        # Fundamental matrix for mass that remains after each local harvest.
        A = np.eye(nt, dtype=np.float64) - (1.0 - h) * Q
        if na:
            release_from_transient = np.linalg.solve(A, (1.0 - h) * R)
        else:
            release_from_transient = np.zeros((nt, 0), dtype=np.float64)

        release = np.zeros((len(tokens), na), dtype=np.float64)
        for t in transient_tokens:
            release[index[int(t)]] = release_from_transient[ti[int(t)]]
        for t in terminal_tokens:
            release[index[int(t)], ai[int(t)]] = 1.0
        harvest_fraction_by_start = 1.0 - np.sum(release, axis=1)
        np.clip(harvest_fraction_by_start, 0.0, 1.0, out=harvest_fraction_by_start)
        # h * total transformed mass = harvested mass, hence this multiplier gives
        # total local reaction flux from an initial unit at each coordinate.
        flux_multiplier_by_start = harvest_fraction_by_start / h
        out_node = np.array([index[int(o)] for o in outs], dtype=np.int32)
        flow = {
            "tokens": tokens,
            "terminal_tokens": terminal_tokens,
            "release": release,
            "harvest_fraction": harvest_fraction_by_start,
            "flux_multiplier": flux_multiplier_by_start,
            "out_node": out_node,
            "transient_set": input_set,
        }
        self._local_flow_cache[gid] = flow
        return flow

    def step(
        self,
        state: PopulationState,
        pool: GenotypePool,
        cfg: ModelConfig,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        n = state.gids.size
        zero_support = np.zeros((n, cfg.basal_functions), dtype=np.float32)
        if n == 0:
            return np.empty(0, dtype=np.float64), zero_support, {
                "active_substrates": 0, "reaction_flux": 0.0, "harvest": 0.0,
                "constructed_mass": 0.0, "environment_mass": 0.0,
                "constructed_resource_flux": 0.0, "constructed_resource_flux_fraction": 0.0,
                "functional_resource_diversity": 0,
                "local_product_flux": 0.0, "local_product_flux_fraction": 0.0,
                "local_metabolic_coordinate_diversity": 0,
                "local_cascade_energy": 0.0,
                "local_cascade_carrier_fraction": 0.0,
                "mean_environmental_function_support": 0.0,
                "environmentally_supported_cell_fraction": 0.0,
                "local_product_function_diversity": 0,
                "mean_local_product_function_support": 0.0,
                "locally_product_supported_cell_fraction": 0.0,
            }

        max_token = pool.max_token
        if max_token > cfg.safety_max_token_id:
            raise RuntimeError("SAFETY_ABORT token_id exceeded")
        self.ensure(max_token)
        self.mass *= (1.0 - cfg.substrate_dissipation)
        source = cfg.source_per_capacity * cfg.nmax
        self.mass[0] += source
        self.total_external_input += source

        gids, inverse, counts = np.unique(state.gids, return_inverse=True, return_counts=True)
        in_parts: List[np.ndarray] = []
        out_parts: List[np.ndarray] = []
        active_gid_parts: List[np.ndarray] = []
        carrier_parts: List[np.ndarray] = []
        module_slices: List[Tuple[int, int, int, int]] = []  # active pos, gid, start, stop
        cursor = 0
        for apos, (gid, c) in enumerate(zip(gids, counts)):
            ia = pool.inputs[int(gid)]
            if ia.size == 0:
                continue
            oa = pool.outputs[int(gid)]
            in_parts.append(ia)
            out_parts.append(oa)
            active_gid_parts.append(np.full(ia.size, apos, dtype=np.int32))
            carrier_parts.append(np.full(ia.size, int(c), dtype=np.float64))
            module_slices.append((apos, int(gid), cursor, cursor + ia.size))
            cursor += ia.size

        per_active_energy = np.zeros(gids.size, dtype=np.float64)
        per_active_local_energy = np.zeros(gids.size, dtype=np.float64)
        total_global_flux = 0.0
        total_local_flux = 0.0
        total_harvest = 0.0
        local_harvest_total = 0.0
        constructed_flux = 0.0
        local_coordinate_tokens: Set[int] = set()

        if in_parts:
            ins = np.concatenate(in_parts).astype(np.int64, copy=False)
            outs = np.concatenate(out_parts).astype(np.int64, copy=False)
            apos_arr = np.concatenate(active_gid_parts).astype(np.int32, copy=False)
            carriers = np.concatenate(carrier_parts).astype(np.float64, copy=False)
            self.ensure(int(max(ins.max(initial=0), outs.max(initial=0))))

            available = self.mass[ins]
            potential = cfg.reaction_rate * carriers * available / (cfg.half_saturation + available)
            demand = np.bincount(ins, weights=potential, minlength=self.mass.size)
            scale = np.ones_like(self.mass)
            pos = demand > self.mass
            if np.any(pos):
                scale[pos] = np.divide(
                    self.mass[pos], demand[pos],
                    out=np.zeros(np.sum(pos), dtype=np.float64), where=demand[pos] > 0
                )
            flux = potential * scale[ins]
            used = np.bincount(ins, weights=flux, minlength=self.mass.size)
            harvested = flux * cfg.harvest_fraction
            nonharvest_product = flux - harvested
            local_retained_total = np.where(
                outs > 0, nonharvest_product * cfg.local_product_retention_fraction, 0.0
            )
            direct_global_product = nonharvest_product - local_retained_total

            env_harvest_by_active = np.bincount(
                apos_arr, weights=harvested, minlength=gids.size
            )
            per_active_energy += np.divide(
                env_harvest_by_active, counts.astype(np.float64),
                out=np.zeros_like(env_harvest_by_active), where=counts > 0
            )
            total_global_flux = float(np.sum(flux))
            total_harvest = float(np.sum(harvested))
            constructed_flux = float(np.sum(flux[(ins > 0) & (flux > 0.0)]))

            release_by_token = np.zeros(self.mass.size, dtype=np.float64)
            tol = 1.0e-15
            locally_active_carriers = 0
            for active_pos, gid, lo, hi in module_slices:
                retained = local_retained_total[lo:hi]
                if retained.size == 0 or float(np.max(retained, initial=0.0)) <= tol:
                    continue
                c = float(counts[active_pos])
                per_carrier = retained / c
                flow = self._local_flow(gid, pool, cfg)
                if flow is None:
                    np.add.at(release_by_token, outs[lo:hi], retained)
                    continue
                initial = np.bincount(
                    flow["out_node"], weights=per_carrier, minlength=len(flow["tokens"])
                ).astype(np.float64, copy=False)
                harvest_pc = float(np.dot(initial, flow["harvest_fraction"]))
                if harvest_pc <= tol:
                    np.add.at(release_by_token, outs[lo:hi], retained)
                    continue
                release_pc = initial @ flow["release"]
                if flow["terminal_tokens"].size:
                    np.add.at(
                        release_by_token,
                        flow["terminal_tokens"],
                        release_pc * c,
                    )
                per_active_local_energy[active_pos] = harvest_pc
                local_h = harvest_pc * c
                local_harvest_total += local_h
                total_local_flux += float(np.dot(initial, flow["flux_multiplier"])) * c
                locally_active_carriers += int(counts[active_pos])
                active_start = (initial > tol) & (flow["harvest_fraction"] > tol)
                if np.any(active_start):
                    local_coordinate_tokens.update(
                        int(x) for x in flow["tokens"][active_start]
                    )

            per_active_energy += per_active_local_energy
            produced_direct = np.bincount(
                outs, weights=direct_global_product, minlength=self.mass.size
            )
            self.mass -= used
            discarded_shared_product = 0.0
            if cfg.shared_ecological_construction_enabled:
                self.mass += produced_direct
                self.mass += release_by_token
            else:
                # ECO_MINUS: preserve reaction uptake, harvest, and within-genotype
                # local routing, but prevent constructed products from becoming a
                # shared ecological state available to later generations/lineages.
                discarded_shared_product = float(np.sum(produced_direct) + np.sum(release_by_token))
            np.maximum(self.mass, 0.0, out=self.mass)
            total_harvest += local_harvest_total
            self.total_harvest += total_harvest
        else:
            locally_active_carriers = 0
            discarded_shared_product = 0.0

        cell_energy = per_active_energy[inverse]
        total_flux = total_global_flux + total_local_flux
        active_substrates = int(np.count_nonzero(self.mass > 1e-9))
        self.max_active_substrates = max(self.max_active_substrates, active_substrates)
        local_carrier_fraction = float(locally_active_carriers / n) if n else 0.0
        return cell_energy, zero_support, {
            "active_substrates": active_substrates,
            "reaction_flux": total_flux,
            "harvest": total_harvest,
            "constructed_mass": float(np.sum(self.mass[1:])),
            "environment_mass": float(np.sum(self.mass)),
            "discarded_shared_product": float(discarded_shared_product),
            "constructed_resource_flux": constructed_flux,
            "constructed_resource_flux_fraction": float(constructed_flux / total_flux) if total_flux > 0 else 0.0,
            "functional_resource_diversity": int(np.count_nonzero(self.mass[1:] > 1e-9)),
            "local_product_flux": total_local_flux,
            "local_product_flux_fraction": float(total_local_flux / total_flux) if total_flux > 0 else 0.0,
            "local_metabolic_coordinate_diversity": int(len(local_coordinate_tokens)),
            "local_cascade_energy": float(local_harvest_total),
            "local_cascade_carrier_fraction": local_carrier_fraction,
            "mean_environmental_function_support": 0.0,
            "environmentally_supported_cell_fraction": 0.0,
            "local_product_function_diversity": int(len(local_coordinate_tokens)),
            "mean_local_product_function_support": 0.0,
            "locally_product_supported_cell_fraction": local_carrier_fraction,
        }


# =============================================================================
# Reproductive-interdependence dynamics
# =============================================================================

def _completeness_from_x(x: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    # x is maintained in [x_min, x_max], so min(clip(x/theta,0,1)) is
    # exactly min(min(x)/theta, 1) without allocating a clipped n x K array.
    q = np.min(x, axis=1).astype(np.float64, copy=False) / cfg.theta
    return np.minimum(q, 1.0)


def _apply_transient_failures_x(
    x: np.ndarray, rng: np.random.Generator, cfg: ModelConfig
) -> Tuple[np.ndarray, np.ndarray]:
    n, k = x.shape
    if n == 0 or cfg.transient_failure_probability <= 0.0:
        return x, np.zeros((n, k), dtype=bool)
    fail = rng.random((n, k)) < cfg.transient_failure_probability
    if not np.any(fail):
        return x, fail
    eff = x.copy()
    eff[fail] = 0.0
    return eff, fail


def _mutate_quantitative_basal(x: np.ndarray, rng: np.random.Generator, cfg: ModelConfig) -> None:
    if x.shape[0] == 0:
        return
    if cfg.mu_x > 0.0:
        m = rng.random(x.shape) < cfg.mu_x
        nm = int(m.sum())
        if nm:
            x[m] += rng.normal(0.0, cfg.sigma_x, nm).astype(np.float32)
            np.clip(x, cfg.x_min, cfg.x_max, out=x)
    if cfg.mu_loss > 0.0:
        lm = rng.random(x.shape) < cfg.mu_loss
        nl = int(lm.sum())
        if nl:
            x[lm] *= rng.uniform(cfg.loss_factor_min, cfg.loss_factor_max, nl).astype(np.float32)


def _pair_effective_expression_per_cell(xa: np.ndarray, xb: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    eta = cfg.regulatory_sharing_efficiency if cfg.functional_interdependence_enabled else 0.0
    total = np.maximum(xa, xb) + (1.0 - eta) * np.minimum(xa, xb)
    return 0.5 * np.mean(total, axis=1)


def _cell_costs(state: PopulationState, pool: GenotypePool, cfg: ModelConfig) -> np.ndarray:
    module_n = pool.module_count_arr[state.gids].astype(np.float64, copy=False)
    return (
        cfg.basal_expression_cost * state.x.mean(axis=1)
        + cfg.reaction_expression_cost * module_n
        + cfg.fusogenicity_cost * state.phi * state.phi
        + cfg.adhesion_cost * state.alpha
    )


def _extract_candidate_pairs(partner: np.ndarray) -> np.ndarray:
    n = partner.size
    if n == 0:
        return np.empty((0, 2), dtype=np.int32)
    idx = np.arange(n, dtype=np.int32)
    left = np.flatnonzero((partner >= 0) & (idx < partner)).astype(np.int32)
    if left.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((left, partner[left].astype(np.int32)))


def _capacity_cull(
    gids: np.ndarray, x: np.ndarray, phi: np.ndarray, alpha: np.ndarray, partner: np.ndarray,
    ids: np.ndarray, ancestors: np.ndarray, origin_pair: np.ndarray,
    rng: np.random.Generator, nmax: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = gids.size
    if n <= nmax:
        return gids, x, phi, alpha, partner, ids, ancestors, origin_pair
    idx = np.arange(n, dtype=np.int32)
    left = np.flatnonzero((partner >= 0) & (idx < partner)).astype(np.int32)
    singles = np.flatnonzero(partner < 0).astype(np.int32)
    reps = np.concatenate((singles, left))
    sizes = np.concatenate((np.ones(singles.size), np.full(left.size, 2.0)))
    keys = -np.log(np.maximum(rng.random(reps.size), np.finfo(float).tiny)) / sizes
    order = np.argsort(keys)
    remove = np.zeros(n, dtype=bool)
    removed = 0
    need = n - nmax
    for u in order:
        r = int(reps[u])
        if partner[r] >= 0:
            j = int(partner[r])
            remove[r] = True; remove[j] = True; removed += 2
        else:
            remove[r] = True; removed += 1
        if removed >= need:
            break
    keep = np.flatnonzero(~remove)
    old2new = np.full(n, -1, dtype=np.int32)
    old2new[keep] = np.arange(keep.size, dtype=np.int32)
    oldp = partner[keep]
    newp = np.full(keep.size, -1, dtype=np.int32)
    valid = oldp >= 0
    if np.any(valid):
        mapped = old2new[oldp[valid]]
        if np.any(mapped < 0):
            raise RuntimeError("capacity cull broke a pair")
        newp[valid] = mapped
    return gids[keep], x[keep], phi[keep], alpha[keep], newp, ids[keep], ancestors[keep], origin_pair[keep]


def _offspring_ancestry(
    parent_indices: np.ndarray,
    state: PopulationState,
    new_ids: np.ndarray,
) -> np.ndarray:
    depth = state.ancestors.shape[1]
    out = np.zeros((parent_indices.size, depth), dtype=np.int64)
    if parent_indices.size == 0 or depth == 0:
        return out
    out[:, 0] = state.individual_id[parent_indices]
    if depth > 1:
        out[:, 1:] = state.ancestors[parent_indices, :-1]
    return out


def one_generation(
    state: PopulationState,
    pool: GenotypePool,
    env: ReactionEnvironment,
    rng: np.random.Generator,
    cfg: ModelConfig,
    next_individual_id: int,
    collect_metrics: bool = True,
) -> Tuple[PopulationState, Dict[str, float], int, int]:
    n = state.gids.size
    if n == 0:
        return state, {"n": 0.0}, next_individual_id, 0
    if n > cfg.safety_max_population:
        raise RuntimeError("SAFETY_ABORT population exceeded")

    # Open-ended reactions provide energetic gain plus two functional channels:
    # realized uptake of constructed inputs and locally retained constructed outputs.
    # Thus both endpoints of the exact hereditary transformation contribute to fitness.
    energy, functional_support, em = env.step(state, pool, cfg)
    intrinsic_effective_x, fail_mask = _apply_transient_failures_x(state.x, rng, cfg)
    effective_x = intrinsic_effective_x + functional_support
    q_ind = _completeness_from_x(effective_x, cfg)
    costs = _cell_costs(state, pool, cfg)
    metabolic = 1.0 + cfg.reaction_energy_bonus * (
        np.maximum(0.0, energy) / (cfg.metabolic_half_energy + np.maximum(0.0, energy))
    )
    clone_rates = cfg.r0 * q_ind * metabolic * np.exp(-costs)

    cand = _extract_candidate_pairs(state.partner)
    if cand.size:
        ia, ib = cand[:, 0], cand[:, 1]
        pkeep = 1.0 - (1.0 - cfg.passive_persistence) * (1.0 - state.alpha[ia]) * (1.0 - state.alpha[ib])
        keep_pair = rng.random(cand.shape[0]) < np.clip(pkeep, 0.0, 1.0)
        persistent = cand[keep_pair]
    else:
        persistent = np.empty((0, 2), dtype=np.int32)

    persistent_member = np.zeros(n, dtype=bool)
    if persistent.size:
        persistent_member[persistent.ravel()] = True
    free = np.flatnonzero(~persistent_member)
    if free.size >= 2:
        perm = free[rng.permutation(free.size)]
        encounters = perm[:2 * (free.size // 2)].reshape(-1, 2)
        ia, ib = encounters[:, 0], encounters[:, 1]
        pfuse = 1.0 - (1.0 - cfg.p0_fusion) * (1.0 - state.phi[ia]) * (1.0 - state.phi[ib])
        fused = rng.random(encounters.shape[0]) < np.clip(pfuse, 0.0, 1.0)
        denovo = encounters[fused]
    else:
        denovo = np.empty((0, 2), dtype=np.int32)

    if persistent.size and denovo.size:
        pairs = np.vstack((persistent, denovo))
        pair_is_persistent = np.concatenate((np.ones(len(persistent), bool), np.zeros(len(denovo), bool)))
    elif persistent.size:
        pairs = persistent
        pair_is_persistent = np.ones(len(persistent), bool)
    else:
        pairs = denovo
        pair_is_persistent = np.zeros(len(denovo), bool)

    active = np.zeros(n, dtype=bool)
    if pairs.size:
        active[pairs.ravel()] = True
    clone_parents = np.flatnonzero(~active)

    pair_rates = np.empty(0, dtype=np.float64)
    genetic_obligate = 0.0
    pair_completeness_mean = 0.0
    mean_genetic_complementation_gain = 0.0
    complementary_locus_fraction = 0.0
    if pairs.size:
        ia, ib = pairs[:, 0], pairs[:, 1]
        if cfg.functional_interdependence_enabled:
            qpair = _completeness_from_x(np.maximum(effective_x[ia], effective_x[ib]), cfg)
        else:
            # Sham-pair control: pair formation and pair-event costs remain, but each
            # member must be individually complete; partner state cannot rescue a loss.
            qpair = np.minimum(q_ind[ia], q_ind[ib])
        pair_completeness_mean = float(np.mean(qpair))

        if collect_metrics:
            qga = _completeness_from_x(state.x[ia], cfg)
            qgb = _completeness_from_x(state.x[ib], cfg)
            qgp = _completeness_from_x(np.maximum(state.x[ia], state.x[ib]), cfg)
            if cfg.functional_interdependence_enabled:
                ggain = np.maximum(0.0, qgp - np.maximum(qga, qgb))
                mean_genetic_complementation_gain = float(np.mean(ggain))
                genetic_obligate = float(np.mean(
                    (qga < 1.0 - 1e-12) & (qgb < 1.0 - 1e-12) & (qgp >= 1.0 - 1e-12)
                ))
                complementary_locus_fraction = float(np.mean(
                    ((state.x[ia] >= cfg.theta) & (state.x[ib] < cfg.theta)) |
                    ((state.x[ib] >= cfg.theta) & (state.x[ia] < cfg.theta))
                ))
            else:
                mean_genetic_complementation_gain = 0.0
                genetic_obligate = 0.0
                complementary_locus_fraction = 0.0

        expr_pair = _pair_effective_expression_per_cell(state.x[ia], state.x[ib], cfg)
        interaction_trait_cost = 0.5 * (
            cfg.fusogenicity_cost * (state.phi[ia] ** 2 + state.phi[ib] ** 2)
            + cfg.adhesion_cost * (state.alpha[ia] + state.alpha[ib])
        )
        reaction_cost = 0.5 * cfg.reaction_expression_cost * (
            pool.module_count_arr[state.gids[ia]] + pool.module_count_arr[state.gids[ib]]
        )
        pair_cost = cfg.basal_expression_cost * expr_pair + interaction_trait_cost + reaction_cost
        pair_metabolic = 0.5 * (metabolic[ia] + metabolic[ib])
        event_cost = np.where(pair_is_persistent, cfg.persistent_pair_cost, cfg.fusion_event_cost)
        pair_rates = cfg.r0 * qpair * pair_metabolic * np.exp(-pair_cost - event_cost)

    expected = float(np.sum(clone_rates[clone_parents]))
    if pair_rates.size:
        expected += float(2.0 * np.sum(pair_rates))
    dscale = min(1.0, cfg.nmax / expected) if expected > 0.0 else 1.0

    if clone_parents.size:
        cc = rng.poisson(clone_rates[clone_parents] * dscale)
        clone_src = np.repeat(clone_parents, cc).astype(np.int32)
    else:
        clone_src = np.empty(0, dtype=np.int32)

    if pairs.size:
        pc = rng.poisson(pair_rates * dscale)
        nz = np.flatnonzero(pc)
        if nz.size:
            pa = np.repeat(pairs[nz, 0], pc[nz]).astype(np.int32)
            pb = np.repeat(pairs[nz, 1], pc[nz]).astype(np.int32)
        else:
            pa = pb = np.empty(0, dtype=np.int32)
    else:
        pa = pb = np.empty(0, dtype=np.int32)
    nprop = pa.size

    n_clone = clone_src.size
    total = n_clone + 2 * nprop
    if total == 0:
        depth = state.ancestors.shape[1]
        empty = PopulationState(
            np.empty(0, np.int32), np.empty((0, cfg.basal_functions), np.float32),
            np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.int32),
            np.empty(0, np.int64), np.empty((0, depth), np.int64),
        )
        return empty, {**em, "n": 0.0}, next_individual_id, 0

    src = np.empty(total, dtype=np.int32)
    origin_pair = np.zeros(total, dtype=bool)
    partner = np.full(total, -1, dtype=np.int32)
    if n_clone:
        src[:n_clone] = clone_src
    if nprop:
        base = n_clone
        src[base:base + 2*nprop:2] = pa
        src[base + 1:base + 2*nprop:2] = pb
        left = np.arange(base, base + 2*nprop, 2, dtype=np.int32)
        partner[left] = left + 1
        partner[left + 1] = left
        origin_pair[base:] = True

    child_gids = mutate_offspring_gids(state.gids[src], pool, rng, cfg)
    child_x = state.x[src].astype(np.float32, copy=True)
    _mutate_quantitative_basal(child_x, rng, cfg)
    child_phi = state.phi[src].astype(np.float32, copy=True)
    child_alpha = state.alpha[src].astype(np.float32, copy=True)
    mphi = rng.random(total) < cfg.mu_phi
    if np.any(mphi):
        child_phi[mphi] += rng.normal(0.0, cfg.sigma_phi, int(np.sum(mphi))).astype(np.float32)
    malpha = rng.random(total) < cfg.mu_alpha
    if np.any(malpha):
        child_alpha[malpha] += rng.normal(0.0, cfg.sigma_alpha, int(np.sum(malpha))).astype(np.float32)
    np.clip(child_phi, 0.0, 1.0, out=child_phi)
    np.clip(child_alpha, 0.0, 1.0, out=child_alpha)

    ids = np.arange(next_individual_id, next_individual_id + total, dtype=np.int64)
    anc = _offspring_ancestry(src, state, ids)
    next_individual_id += total

    child_gids, child_x, child_phi, child_alpha, partner, ids, anc, origin_pair = _capacity_cull(
        child_gids, child_x, child_phi, child_alpha, partner, ids, anc, origin_pair, rng, cfg.nmax
    )
    nn = child_gids.size

    if collect_metrics:
        # Counterfactual genetic isolated replacement: no transient failure and no partner sharing.
        qg = _completeness_from_x(state.x, cfg)
        baseline_isolated_rates = cfg.r0 * qg * np.exp(-costs)
        isolated_rates = baseline_isolated_rates * metabolic
        # V11 reaction products no longer project onto the finite K basal vector.
        # Their causal contribution is exact-token metabolic throughput, measured by
        # the reproduction-rate increase relative to the same inherited state with
        # reaction-derived energy removed.
        reaction_metabolic_gain = np.maximum(0.0, isolated_rates - baseline_isolated_rates)
        reaction_energy_rescued = (baseline_isolated_rates < 1.0) & (isolated_rates >= 1.0)
        qeco = qg
        ecological_isolated_rates = isolated_rates
        ecological_gain = np.zeros_like(qg)
        environmentally_rescued = np.zeros(n, dtype=bool)
        reaction_function_dependent = reaction_metabolic_gain > 1e-12
        autonomous = np.all(state.x >= cfg.theta, axis=1)
        metrics = {
            **em,
            "n": int(nn),
            "structural_mutation_events_cumulative": int(pool.structural_mutation_events),
            "reaction_identities_ever": int(len(pool.component_identities_ever)),
            "pair_birth_fraction": float(np.mean(origin_pair)) if nn else 0.0,
            "born_paired_fraction": float(np.mean(state.partner >= 0)) if n else 0.0,
            "persistent_interaction_fraction": float(2 * len(persistent) / n),
            "de_novo_fusion_fraction": float(2 * len(denovo) / n),
            "realized_pair_interaction_fraction": float(2 * len(pairs) / n),
            "pair_completeness_mean": pair_completeness_mean,
            "genetic_obligate_complementation_fraction": genetic_obligate,
            "mean_genetic_complementation_gain": mean_genetic_complementation_gain,
            "complementary_locus_fraction": complementary_locus_fraction,
            "genetic_isolated_replacement": float(np.mean(isolated_rates)),
            "autonomous_cell_fraction": float(np.mean(autonomous)),
            "genetic_self_sustaining_cell_fraction": float(np.mean(isolated_rates >= 1.0)),
            "mean_genetic_reproductive_completeness": float(np.mean(qg)),
            "mean_ecologically_supported_completeness": float(np.mean(qeco)),
            "mean_ecological_completeness_gain": float(np.mean(ecological_gain)),
            "environmentally_rescued_cell_fraction": float(np.mean(environmentally_rescued)),
            "reaction_function_dependent_cell_fraction": float(np.mean(reaction_function_dependent)),
            "reaction_metabolic_gain": float(np.mean(reaction_metabolic_gain)),
            "reaction_energy_rescued_cell_fraction": float(np.mean(reaction_energy_rescued)),
            "ecologically_supported_isolated_replacement": float(np.mean(ecological_isolated_rates)),
            "specialization_index": float(np.mean(state.x < cfg.theta)),
            "locus_diversity": float(np.mean(np.std(state.x, axis=0, ddof=0))) if n > 1 else 0.0,
            "mean_functional_investment": float(np.mean(state.x)),
            "mean_phi": float(np.mean(state.phi)),
            "mean_alpha": float(np.mean(state.alpha)),
            "mean_reaction_modules": float(np.mean(pool.module_count_arr[state.gids])) if n else 0.0,
            "max_reaction_modules": int(np.max(pool.module_count_arr[state.gids])) if n else 0,
            "functional_genotype_diversity": int(np.unique(state.gids).size),
            "density_scale": float(dscale),
        }
    else:
        metrics = {"n": int(nn), "density_scale": float(dscale)}
    return PopulationState(child_gids, child_x, child_phi, child_alpha, partner, ids, anc), metrics, next_individual_id, int(nn)

def validate_state(state: PopulationState, pool: GenotypePool, cfg: ModelConfig) -> None:
    n = state.gids.size
    if not (state.phi.size == state.alpha.size == state.partner.size == state.individual_id.size == n):
        raise RuntimeError("state array length mismatch")
    if state.x.shape != (n, cfg.basal_functions):
        raise RuntimeError("quantitative basal array shape mismatch")
    if state.ancestors.shape != (n, cfg.modes_filter_generations):
        raise RuntimeError("ancestry ring shape mismatch")
    if n > cfg.nmax or n > cfg.safety_max_population:
        raise RuntimeError("population exceeded physical/safety capacity")
    if n:
        if int(state.gids.min()) < 0 or int(state.gids.max()) >= len(pool.keys):
            raise RuntimeError("invalid genotype id")
        if not np.all(np.isfinite(state.x)) or not np.all(np.isfinite(state.phi)) or not np.all(np.isfinite(state.alpha)):
            raise RuntimeError("non-finite quantitative/interaction trait")
        if np.any(state.x < cfg.x_min) or np.any(state.x > cfg.x_max):
            raise RuntimeError("quantitative basal investment out of bounds")
        linked = np.flatnonzero(state.partner >= 0)
        if linked.size:
            if np.any(state.partner[linked] >= n) or np.any(state.partner[linked] == linked):
                raise RuntimeError("invalid partner link")
            if np.any(state.partner[state.partner[linked]] != linked):
                raise RuntimeError("non-reciprocal partner link")


# =============================================================================
# Channon standard activity benchmark -- same implementation logic as 11_STANDARD
# =============================================================================

class ChannonActivity:
    def __init__(self):
        self.basic_history: Dict[str, int] = defaultdict(int)
        self.normalized_history: Dict[str, int] = defaultdict(int)
        self._raw_rows: List[Dict[str, Any]] = []
        self._norm_present: List[Dict[str, float]] = []
        self.rows: List[Dict[str, Any]] = []

    def observe(
        self, snapshot_index: int, step: int, real_components: Set[str],
        shadow_components: Set[str], analysis_births: int,
    ) -> Dict[str, Any]:
        for c in real_components:
            self.basic_history[c] += 1
        for c in real_components | shadow_components:
            self.normalized_history[c] += int(c in real_components) - int(c in shadow_components)
        d_real = len(real_components)
        basic = [self.basic_history[c] for c in real_components]
        norm_map = {c: float(self.normalized_history[c]) for c in real_components}
        norm = list(norm_map.values())
        a_cum = float(sum(basic))
        row = {
            "snapshot": int(snapshot_index), "step": int(step), "analysis_births": int(analysis_births),
            "component_diversity_D_R": int(d_real),
            "A_cum": a_cum,
            "A_mean_cum": a_cum / d_real if d_real else 0.0,
            "A_median_cum": float(statistics.median(basic)) if basic else 0.0,
            "AN_cum": float(sum(norm)),
            "AN_mean_cum": float(sum(norm)) / d_real if d_real else 0.0,
            "AN_median_cum": float(statistics.median(norm)) if norm else 0.0,
            "real_component_count": len(real_components),
            "shadow_component_count": len(shadow_components),
        }
        self._raw_rows.append(row)
        self._norm_present.append(norm_map)
        return row

    def minimum_normalized_activity(self) -> float:
        m = 0.0
        for row in self._norm_present:
            if row:
                m = min(m, min(float(v) for v in row.values()))
        return float(m)

    def finalize(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if threshold is None:
            threshold = abs(self.minimum_normalized_activity())
        threshold = float(threshold)
        ever_new: Set[str] = set()
        running_sum = 0.0
        out = []
        for idx, (base, m) in enumerate(zip(self._raw_rows, self._norm_present), start=1):
            d_real = int(base["component_diversity_D_R"])
            adaptive = [c for c, a in m.items() if a > threshold]
            new = [c for c in adaptive if c not in ever_new]
            an_new = (sum(m[c] for c in new) / d_real) if d_real else 0.0
            ever_new.update(new)
            running_sum += float(an_new)
            row = dict(base)
            row.update({
                "adaptive_threshold": threshold,
                "adaptive_component_diversity": len(adaptive),
                "new_adaptive_components": len(new),
                "AN_new": float(an_new),
                "AN_new_running_mean": float(running_sum / idx),
            })
            out.append(row)
        self.rows = out
        return out


class NeutralShadow:
    """Generational neutral shadow with the same mutation operator and snapshot reset.

    The shadow uses its own intern pool, so shadow-only genotypes/tokens can never
    alter the real run. Parent selection is uniform. Structural mutation is the
    same event-driven implementation used by real offspring.
    """
    def __init__(self, cfg: ModelConfig, seed: int):
        self.cfg = cfg
        self.rng_parent = np.random.default_rng(stable_seed(seed, "shadow_parent"))
        self.rng_mut = np.random.default_rng(stable_seed(seed, "shadow_mut"))
        self.pool = GenotypePool(cfg.basal_functions)
        self.gids = np.empty(0, dtype=np.int32)

    def reset_from_real(self, state: PopulationState, pool: GenotypePool) -> None:
        # A fresh intern dictionary guarantees that no shadow-only historical state
        # affects later mutation bookkeeping after Channon's required snapshot reset.
        self.pool = GenotypePool(self.cfg.basal_functions)
        if state.gids.size == 0:
            self.gids = np.empty(0, dtype=np.int32)
            return
        mapping: Dict[int, int] = {}
        out = np.empty(state.gids.size, dtype=np.int32)
        for rgid in np.unique(state.gids):
            sgid = self.pool.intern(pool.key(int(rgid)))
            mapping[int(rgid)] = int(sgid)
        # n is small; vectorized search avoids one Python mutation operation per cell.
        for rgid, sgid in mapping.items():
            out[state.gids == rgid] = sgid
        self.gids = out

    def reproduce_to_size(self, births: int) -> None:
        births = int(births)
        if births <= 0 or self.gids.size == 0:
            self.gids = np.empty(0, dtype=np.int32)
            return
        parent_idx = self.rng_parent.integers(0, self.gids.size, size=births)
        parent_gids = self.gids[parent_idx]
        self.gids = mutate_offspring_gids(parent_gids, self.pool, self.rng_mut, self.cfg)

    def components(self) -> Set[str]:
        out: Set[str] = set()
        for gid in np.unique(self.gids):
            out.update(self.pool.components(int(gid)))
        return out


# =============================================================================
# Published boundedness comparison used by the standard OEE benchmark
# =============================================================================

def _bic(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), 1e-15)
    return n * math.log(rss / n) + k * math.log(n)


def fit_published_boundedness(x_in: Sequence[float], y_in: Sequence[float]) -> Dict[str, Any]:
    pairs = [(float(x), float(y)) for x, y in zip(x_in, y_in)
             if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 4:
        return {"valid": False, "reason": "fewer_than_4_points"}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    x = x - float(x.min())
    span = float(x.max())
    if span <= 0.0:
        return {"valid": False, "reason": "zero_time_span"}
    t = x / span
    y0 = float(y[0]); g = y - y0
    if float(np.max(g) - np.min(g)) <= 1e-12:
        return {
            "valid": True, "n": len(y), "best_model": "constant", "best_family": "bounded",
            "best_bic": _bic(float(np.sum(g*g)), len(y), 1), "best_params": {"baseline": y0},
            "delta_bic_bounded_minus_unbounded": None, "models": [],
        }

    def bounded_model(tt, a, b):
        return a * tt / (tt + b)
    def unbounded_model(tt, a, b):
        return np.power(1.0 + b * tt, a) - 1.0

    models: List[Dict[str, Any]] = []
    bounded_error = None; unbounded_error = None
    a0 = max(float(np.max(g)), 1e-6)
    try:
        popt, _ = curve_fit(
            bounded_model, t, g, p0=(a0, 0.5),
            bounds=([0.0, 1e-9], [np.inf, np.inf]), maxfev=50_000,
        )
        pred = bounded_model(t, *popt); rss = float(np.sum((g - pred) ** 2))
        models.append({
            "model": "rectangular_hyperbola", "family": "bounded", "rss": rss,
            "bic": _bic(rss, len(y), 2),
            "params": {"a": float(popt[0]), "b": float(popt[1]), "baseline": y0},
        })
    except Exception as e:
        bounded_error = str(e)
    try:
        best = None
        for p0 in ((1.0, 1.0), (0.5, max(a0, 1.0)), (2.0, max(a0, 1.0))):
            try:
                popt, _ = curve_fit(
                    unbounded_model, t, g, p0=p0,
                    bounds=([1e-9, 1e-9], [10.0, 1e9]), maxfev=50_000,
                )
                pred = unbounded_model(t, *popt); rss = float(np.sum((g - pred) ** 2))
                rec = {
                    "model": "power_law", "family": "unbounded", "rss": rss,
                    "bic": _bic(rss, len(y), 2),
                    "params": {"a": float(popt[0]), "b": float(popt[1]), "baseline": y0},
                }
                if best is None or rec["rss"] < best["rss"]:
                    best = rec
            except Exception:
                pass
        if best is not None:
            models.append(best)
        else:
            unbounded_error = "all power-law fits failed"
    except Exception as e:
        unbounded_error = str(e)
    if not models:
        return {"valid": False, "reason": "both published model fits failed",
                "bounded_error": bounded_error, "unbounded_error": unbounded_error}
    models.sort(key=lambda r: (r["bic"], r["rss"]))
    best = models[0]
    bb = next((m["bic"] for m in models if m["family"] == "bounded"), None)
    ub = next((m["bic"] for m in models if m["family"] == "unbounded"), None)
    delta = float(bb) - float(ub) if bb is not None and ub is not None else None
    return {
        "valid": True, "n": len(y), "best_model": best["model"], "best_family": best["family"],
        "best_bic": best["bic"], "best_params": best["params"],
        "delta_bic_bounded_minus_unbounded": delta, "models": models,
    }


# =============================================================================
# Online MODES persistence filter
# =============================================================================

def functional_genotype_key(pool: GenotypePool, gid: int) -> str:
    return json.dumps(pool.components(int(gid)), separators=(",", ":"))


def genotype_len(key: str) -> int:
    return len(json.loads(key))


@dataclass
class PendingModesSnapshot:
    snapshot: int
    generation: int
    analysis_births: int
    genotype_counts: Dict[str, int]
    genotype_carriers: Dict[str, List[int]]


class OnlineModes:
    def __init__(self, filter_generations: int):
        self.t = int(filter_generations)
        self.pending: Deque[PendingModesSnapshot] = deque()
        self.previous_filtered: Set[str] = set()
        self.permanent_history: Set[str] = set()
        self.cumulative_change = 0
        self.cumulative_novelty = 0
        self.rows: List[Dict[str, Any]] = []

    def add_snapshot(self, snap: PendingModesSnapshot) -> None:
        self.pending.append(snap)

    def evaluate_due(self, current_generation: int, state: PopulationState) -> None:
        if self.t <= 0 or not self.pending or current_generation < self.pending[0].generation + self.t:
            return
        ancestor_ids = set(int(x) for x in state.ancestors[:, self.t - 1]) if state.gids.size else set()
        ancestor_ids.discard(0)
        while self.pending and current_generation >= self.pending[0].generation + self.t:
            s = self.pending.popleft()
            filtered: Set[str] = set()
            for gkey, ids in s.genotype_carriers.items():
                if any(int(i) in ancestor_ids for i in ids):
                    filtered.add(gkey)
            change = sum(1 for c in filtered if c not in self.previous_filtered)
            novelty = sum(1 for c in filtered if c not in self.permanent_history)
            self.cumulative_change += change; self.cumulative_novelty += novelty
            neval = len(self.rows) + 1
            complexity = max((genotype_len(c) for c in filtered), default=0)
            total = sum(s.genotype_counts.get(c, 0) for c in filtered)
            ecology = 0.0
            if total > 0:
                for c in filtered:
                    p = s.genotype_counts.get(c, 0) / total
                    if p > 0.0:
                        ecology -= p * math.log2(p)
            self.rows.append({
                "filter_generations": self.t, "snapshot": s.snapshot, "step": s.generation,
                "analysis_births": s.analysis_births, "filtered_components": len(filtered),
                "change": change, "novelty": novelty,
                "cumulative_change": self.cumulative_change,
                "cumulative_novelty": self.cumulative_novelty,
                "change_running_mean": self.cumulative_change / neval,
                "novelty_running_mean": self.cumulative_novelty / neval,
                "complexity": complexity, "ecology": ecology,
            })
            self.previous_filtered = filtered
            self.permanent_history.update(filtered)


# =============================================================================
# Run simulation
# =============================================================================

def initial_state(cfg: ModelConfig, pool: GenotypePool) -> Tuple[PopulationState, int]:
    all_mask = (1 << cfg.basal_functions) - 1
    # One ancestral reaction; all additional transformations must arise by mutation.
    gid0 = pool.intern((all_mask, ((0, 1),)))
    n = cfg.n0
    ids = np.arange(1, n + 1, dtype=np.int64)
    return PopulationState(
        gids=np.full(n, gid0, dtype=np.int32),
        x=np.full((n, cfg.basal_functions), cfg.initial_x, dtype=np.float32),
        phi=np.zeros(n, dtype=np.float32),
        alpha=np.zeros(n, dtype=np.float32),
        partner=np.full(n, -1, dtype=np.int32),
        individual_id=ids,
        ancestors=np.zeros((n, cfg.modes_filter_generations), dtype=np.int64),
    ), n + 1


def real_components(state: PopulationState, pool: GenotypePool) -> Set[str]:
    out: Set[str] = set()
    for gid in np.unique(state.gids):
        out.update(pool.components(int(gid)))
    return out


def modes_snapshot(
    snapshot: int, generation: int, births: int, state: PopulationState, pool: GenotypePool
) -> PendingModesSnapshot:
    counts: Dict[str, int] = defaultdict(int)
    carriers: Dict[str, List[int]] = defaultdict(list)
    for gid in np.unique(state.gids):
        idx = np.flatnonzero(state.gids == gid)
        key = functional_genotype_key(pool, int(gid))
        counts[key] = int(idx.size)
        carriers[key] = [int(x) for x in state.individual_id[idx]]
    return PendingModesSnapshot(snapshot, generation, births, dict(counts), dict(carriers))



def _copy_state(state: PopulationState) -> PopulationState:
    return PopulationState(
        state.gids.copy(), state.x.copy(), state.phi.copy(), state.alpha.copy(),
        state.partner.copy(), state.individual_id.copy(), state.ancestors.copy(),
    )


def _assay_env_clone(env: ReactionEnvironment, cfg: ModelConfig) -> ReactionEnvironment:
    out = ReactionEnvironment(cfg)
    out.mass = env.mass.copy()
    out.total_external_input = float(env.total_external_input)
    out.total_harvest = float(env.total_harvest)
    out.max_active_substrates = int(env.max_active_substrates)
    return out


def _component_tuple(component: str) -> Tuple[int, int]:
    text = str(component)
    if not text.startswith("R") or ">" not in text:
        raise ValueError(f"invalid reaction component: {component}")
    a, b = text[1:].split(">", 1)
    return int(a), int(b)


def _remap_active_state_for_assay(
    state: PopulationState, pool: GenotypePool, cfg: ModelConfig,
    knockout: Optional[Tuple[int, int]] = None,
) -> Tuple[PopulationState, GenotypePool]:
    """Build a compact assay-only pool containing active genotypes.

    If knockout is supplied, every occurrence of that exact hereditary transformation
    is removed from current carriers.  Basal investment, interaction traits, partner
    state, individual IDs and ancestry are held fixed.
    """
    apool = GenotypePool(cfg.basal_functions)
    mapping: Dict[int, int] = {}
    ka, kb = knockout if knockout is not None else (-1, -1)
    for old in np.unique(state.gids):
        mask, modules = pool.key(int(old))
        if knockout is not None:
            modules = tuple((a,b) for a,b in modules if not (int(a)==ka and int(b)==kb))
        mapping[int(old)] = apool.intern((mask, tuple(modules)))
    gids = np.fromiter((mapping[int(g)] for g in state.gids), dtype=np.int32, count=state.gids.size)
    out = _copy_state(state)
    out.gids = gids
    return out, apool


def _probe_cfg(cfg: ModelConfig) -> ModelConfig:
    """Mutation-free paired intervention; ecological/reproductive rules stay unchanged."""
    return replace(
        cfg,
        mu_x=0.0, mu_loss=0.0,
        mu_module_point_per_module=0.0, mu_module_dup_per_module=0.0,
        mu_module_del_per_module=0.0, mu_module_add=0.0,
        mu_phi=0.0, mu_alpha=0.0,
    )


def _tagged_children(child: PopulationState, carrier_ids: np.ndarray) -> int:
    if child.gids.size == 0 or carrier_ids.size == 0:
        return 0
    # one-generation probe: col0 is the exact parent ID
    return int(np.count_nonzero(np.isin(child.ancestors[:, 0], carrier_ids, assume_unique=False)))


def _exact_signflip_p_greater(diffs: Sequence[float]) -> float:
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return 1.0
    obs = float(np.mean(d))
    if obs <= 0.0 or np.all(np.abs(d) < 1e-15):
        return 1.0
    # Full scientific mode uses 8 repeats -> 256 exact sign assignments.
    ge = 0
    total = 1 << n
    for mask in range(total):
        signs = np.ones(n, dtype=float)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0
        if float(np.mean(signs * d)) >= obs - 1e-15:
            ge += 1
    return float(ge / total)


def causal_component_probe(
    component: str, generation: int, state: PopulationState, pool: GenotypePool,
    env: ReactionEnvironment, cfg: ModelConfig, root_seed: int,
) -> Dict[str, Any]:
    target = _component_tuple(component)
    carrier = np.zeros(state.gids.size, dtype=bool)
    for gid in np.unique(state.gids):
        if target in pool.key(int(gid))[1]:
            carrier[state.gids == gid] = True
    carrier_ids = state.individual_id[carrier].copy()
    if carrier_ids.size == 0:
        return {
            "component": component, "generation": int(generation), "n_carriers": 0,
            "mean_log_descendant_effect": 0.0, "mean_descendant_difference": 0.0,
            "p_value": 1.0, "repeats": int(cfg.causal_probe_repeats),
            "intact_descendants_json": "[]", "knockout_descendants_json": "[]",
        }

    pcfg = _probe_cfg(cfg)
    intact_state0, intact_pool = _remap_active_state_for_assay(state, pool, pcfg, None)
    ko_state0, ko_pool = _remap_active_state_for_assay(state, pool, pcfg, target)
    intact_counts: List[int] = []
    ko_counts: List[int] = []
    log_effects: List[float] = []
    for r in range(int(cfg.causal_probe_repeats)):
        seed = stable_seed(root_seed, "causal_probe", generation, component, r)
        ri = np.random.default_rng(seed)
        rk = np.random.default_rng(seed)
        si = _copy_state(intact_state0)
        sk = _copy_state(ko_state0)
        ei = _assay_env_clone(env, pcfg)
        ek = _assay_env_clone(env, pcfg)
        next_id = int(max(int(state.individual_id.max(initial=0)) + 1, 1))
        ci, _, _, _ = one_generation(si, intact_pool, ei, ri, pcfg, next_id, collect_metrics=False)
        ck, _, _, _ = one_generation(sk, ko_pool, ek, rk, pcfg, next_id, collect_metrics=False)
        ni = _tagged_children(ci, carrier_ids)
        nk = _tagged_children(ck, carrier_ids)
        intact_counts.append(ni); ko_counts.append(nk)
        log_effects.append(float(math.log1p(ni) - math.log1p(nk)))
    diffs = np.asarray(intact_counts, dtype=float) - np.asarray(ko_counts, dtype=float)
    return {
        "component": component, "generation": int(generation), "n_carriers": int(carrier_ids.size),
        "mean_log_descendant_effect": float(np.mean(log_effects)),
        "mean_descendant_difference": float(np.mean(diffs)),
        "p_value": _exact_signflip_p_greater(log_effects),
        "repeats": int(cfg.causal_probe_repeats),
        "intact_descendants_json": json.dumps(intact_counts),
        "knockout_descendants_json": json.dumps(ko_counts),
    }


def _bh_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(list(pvals), dtype=float)
    n = int(p.size)
    if n == 0:
        return []
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank0 in range(n-1, -1, -1):
        idx = int(order[rank0])
        rank = rank0 + 1
        val = min(prev, float(p[idx]) * n / rank)
        q[idx] = min(1.0, val)
        prev = val
    return q.tolist()


def finalize_causal_rows(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in bundle.get("causal_raw", [])]
    if not rows:
        return []
    qs = _bh_adjust([float(r.get("p_value", 1.0)) for r in rows])
    alpha = float(bundle.get("meta", {}).get("causal_probe_alpha", 0.05))
    for r, q in zip(rows, qs):
        effect_positive = float(r.get("mean_log_descendant_effect", 0.0)) > 0.0
        p = float(r.get("p_value", 1.0))
        r["q_value_BH_within_run"] = float(q)
        # Evidence tiers are reported in parallel.  None is promoted to a universal
        # OEE gate: effect sign is the direct causal observation, while p and BH-q
        # quantify progressively stronger statistical support.
        r["positive_causal_effect"] = int(effect_positive)
        r["nominal_p_le_0_05"] = int(effect_positive and p <= alpha)
        r["BH_q_le_0_05"] = int(effect_positive and float(q) <= alpha)
        # Backward-compatible field, explicitly the strongest statistical tier only.
        r["causally_adaptive_BH_confirmed"] = r["BH_q_le_0_05"]
    return rows


def causal_hallmark_timeseries(bundle: Mapping[str, Any], finalized: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    snaps = [dict(r) for r in bundle.get("component_snapshots", [])]
    tier_fields = {
        "positive": "positive_causal_effect",
        "nominal": "nominal_p_le_0_05",
        "BH": "BH_q_le_0_05",
    }
    adaptive_at: Dict[str, Dict[str, int]] = {}
    for tier, field in tier_fields.items():
        adaptive_at[tier] = {
            str(r["component"]): int(r["generation"])
            for r in finalized if int(r.get(field, 0)) == 1
        }
    out: List[Dict[str, Any]] = []
    running_max = {k: 0 for k in tier_fields}
    prev_cum = {k: 0 for k in tier_fields}
    prev_births = 0
    for s in snaps:
        gen = int(s["generation"]); births = int(s.get("analysis_births", 0))
        current = set(str(x) for x in s.get("components", []))
        row: Dict[str, Any] = {
            "snapshot": int(s["snapshot"]), "generation": gen, "analysis_births": births,
        }
        db = max(0, births - prev_births)
        for tier in tier_fields:
            validated = {c for c,g in adaptive_at[tier].items() if g <= gen}
            current_adaptive = current & validated
            cumulative = len(validated)
            new = max(0, cumulative - prev_cum[tier])
            rate = 1000.0 * new / db if db > 0 else 0.0
            running_max[tier] = max(running_max[tier], len(current_adaptive))
            row.update({
                f"current_{tier}_causal_component_diversity": int(len(current_adaptive)),
                f"running_max_{tier}_causal_component_diversity": int(running_max[tier]),
                f"cumulative_{tier}_causal_innovations": int(cumulative),
                f"new_{tier}_causal_innovations": int(new),
                f"new_{tier}_causal_innovations_per_1000_births": float(rate),
            })
            prev_cum[tier] = cumulative
        out.append(row)
        prev_births = births
    return out


def aggregate_behavioral_hallmarks(bundles: Sequence[Mapping[str, Any]], out_dir: Path) -> Dict[str, Any]:
    assay_rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    ecological_rows: List[Dict[str, Any]] = []
    for b in bundles:
        meta = b.get("meta", {})
        rid = str(meta.get("run_id"))
        finalized = finalize_causal_rows(b)
        for r in finalized:
            z = dict(r); z["run_id"] = rid; z["seed"] = meta.get("seed"); assay_rows.append(z)
        ts = causal_hallmark_timeseries(b, finalized)
        for r in ts:
            z = dict(r); z["run_id"] = rid; z["seed"] = meta.get("seed"); ts_rows.append(z)
        requested = int(meta.get("generations_requested", 0) or 0)
        half = requested / 2.0
        late = [r for r in ts if float(r["generation"]) >= half]
        early = [r for r in ts if float(r["generation"]) < half]
        summary: Dict[str, Any] = {
            "run_id": rid, "completed": meta.get("completed"), "extinct": meta.get("extinct"),
            "n_causal_assays": len(finalized),
            "n_positive_causal_components": sum(int(r.get("positive_causal_effect",0)) for r in finalized),
            "n_nominal_p05_causal_components": sum(int(r.get("nominal_p_le_0_05",0)) for r in finalized),
            "n_BH_q05_causal_components": sum(int(r.get("BH_q_le_0_05",0)) for r in finalized),
            "scope": "finite-horizon behavioral evidence; effect/p/q tiers are reported separately and are not collapsed into an OEE score",
        }
        for tier in ("positive","nominal","BH"):
            new_key=f"new_{tier}_causal_innovations"
            max_key=f"running_max_{tier}_causal_component_diversity"
            late_new = int(sum(int(r.get(new_key,0)) for r in late))
            late_births = 0
            if len(late) >= 2:
                late_births=max(0,int(late[-1]["analysis_births"])-int(late[0]["analysis_births"]))
            late_rate=1000.0*late_new/late_births if late_births>0 else 0.0
            early_max=max((int(r.get(max_key,0)) for r in early),default=0)
            final_max=max((int(r.get(max_key,0)) for r in ts),default=0)
            summary.update({
                f"late_{tier}_causal_innovations":late_new,
                f"late_{tier}_causal_novelty_per_1000_births":late_rate,
                f"early_half_running_max_{tier}_causal_diversity":early_max,
                f"final_running_max_{tier}_causal_diversity":final_max,
                f"ongoing_{tier}_causal_novelty_observed_over_late_half":int(late_new>0),
                f"{tier}_causal_diversity_growth_observed":int(final_max>early_max),
            })
        summary_rows.append(summary)
        traj = [dict(r) for r in b.get("trajectory", [])]
        if traj:
            e = [r for r in traj if float(r.get("generation",0)) < half]
            l = [r for r in traj if float(r.get("generation",0)) >= half]
            def mx(rows, key): return max((float(r.get(key,0.0) or 0.0) for r in rows), default=0.0)
            ecological_rows.append({
                "run_id": rid,
                "early_max_component_diversity": mx(e,"component_diversity"),
                "late_max_component_diversity": mx(l,"component_diversity"),
                "early_max_functional_resource_diversity": mx(e,"functional_resource_diversity"),
                "late_max_functional_resource_diversity": mx(l,"functional_resource_diversity"),
                "early_max_local_metabolic_coordinate_diversity": mx(e,"local_metabolic_coordinate_diversity"),
                "late_max_local_metabolic_coordinate_diversity": mx(l,"local_metabolic_coordinate_diversity"),
                "early_max_genotype_diversity": mx(e,"functional_genotype_diversity"),
                "late_max_genotype_diversity": mx(l,"functional_genotype_diversity"),
                "note": "descriptive ecological expansion; no composite score",
            })
    write_csv(out_dir / "07_CAUSAL_ADAPTIVE_NOVELTY_ASSAYS.csv", assay_rows)
    write_csv(out_dir / "08_BEHAVIORAL_HALLMARK_TIMESERIES.csv", ts_rows)
    write_csv(out_dir / "09_OEE_HALLMARK_SUMMARY.csv", summary_rows)
    write_csv(out_dir / "09B_ECOLOGICAL_EXPANSION_SUMMARY.csv", ecological_rows)
    return {
        "n_runs": len(bundles),
        "positive_effect_evidence": {
            "total_components": sum(int(r["n_positive_causal_components"]) for r in summary_rows),
            "runs_with_late_novelty": sum(int(r["ongoing_positive_causal_novelty_observed_over_late_half"]) for r in summary_rows),
            "runs_with_diversity_growth": sum(int(r["positive_causal_diversity_growth_observed"]) for r in summary_rows),
        },
        "nominal_p05_evidence": {
            "total_components": sum(int(r["n_nominal_p05_causal_components"]) for r in summary_rows),
            "runs_with_late_novelty": sum(int(r["ongoing_nominal_causal_novelty_observed_over_late_half"]) for r in summary_rows),
            "runs_with_diversity_growth": sum(int(r["nominal_causal_diversity_growth_observed"]) for r in summary_rows),
        },
        "BH_q05_evidence": {
            "total_components": sum(int(r["n_BH_q05_causal_components"]) for r in summary_rows),
            "runs_with_late_novelty": sum(int(r["ongoing_BH_causal_novelty_observed_over_late_half"]) for r in summary_rows),
            "runs_with_diversity_growth": sum(int(r["BH_causal_diversity_growth_observed"]) for r in summary_rows),
        },
        "interpretation": "direct causal effect, nominal p and BH-q are parallel evidence tiers; no tier is a universal OEE gate",
    }


def aggregate_hallmark_scaling(bundles: Sequence[Mapping[str, Any]], out_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for b in bundles:
        cap = int(b.get("meta", {}).get("nmax", 0))
        f = finalize_causal_rows(b)
        ts = causal_hallmark_timeseries(b, f)
        rec = {
            "run_id": b.get("meta",{}).get("run_id"), "nmax": cap,
            "completed": b.get("meta",{}).get("completed"),
            "max_causally_adaptive_diversity": max((int(r.get("running_max_positive_causal_component_diversity",0)) for r in ts), default=0),
            "cumulative_causally_adaptive_innovations": max((int(r.get("cumulative_positive_causal_innovations",0)) for r in ts), default=0),
            "max_component_diversity": max((int(r.get("component_diversity",0)) for r in b.get("trajectory",[])), default=0),
            "max_local_metabolic_coordinate_diversity": max((int(r.get("local_metabolic_coordinate_diversity",0)) for r in b.get("trajectory",[])), default=0),
        }
        rows.append(rec); grouped[cap].append(rec)
    summary=[]
    for cap in sorted(grouped):
        rr=grouped[cap]
        summary.append({
            "nmax":cap, "n_runs":len(rr),
            "max_observed_causally_adaptive_diversity":max(int(x["max_causally_adaptive_diversity"]) for x in rr),
            "median_run_max_causally_adaptive_diversity":float(np.median([int(x["max_causally_adaptive_diversity"]) for x in rr])),
            "max_cumulative_causally_adaptive_innovations":max(int(x["cumulative_causally_adaptive_innovations"]) for x in rr),
            "max_component_diversity":max(int(x["max_component_diversity"]) for x in rr),
            "max_local_metabolic_coordinate_diversity":max(int(x["max_local_metabolic_coordinate_diversity"]) for x in rr),
        })
    write_csv(out_dir / "09C_HALLMARK_SCALABILITY_BY_RUN.csv", rows)
    write_csv(out_dir / "09D_HALLMARK_SCALABILITY_SUMMARY.csv", summary)
    return {"by_capacity": summary, "scope":"physical-scale evidence reported directly; no single pass/fail threshold"}


def simulate_task(task: RunTask) -> Dict[str, Any]:
    t0 = time.time()
    cfg = task.cfg
    rng = np.random.default_rng(task.seed)
    pool = GenotypePool(cfg.basal_functions)
    state, next_id = initial_state(cfg, pool)
    env = ReactionEnvironment(cfg)
    shadow = NeutralShadow(cfg, task.seed)
    shadow.reset_from_real(state, pool)
    activity = ChannonActivity()
    modes = OnlineModes(cfg.modes_filter_generations)

    causal_raw: List[Dict[str, Any]] = []
    component_snapshots: List[Dict[str, Any]] = []
    assayed_components: Set[str] = set()
    prev_components: Set[str] = set()
    ancestral_components = real_components(state, pool)

    analysis_births = 0
    snapshot_index = 0
    trajectory: List[Dict[str, Any]] = []
    max_pop = state.gids.size
    max_modules = 1
    max_function_diversity = len(real_components(state, pool))
    max_functional_genotype_diversity = 1
    extinct = False
    safety_stop = ""
    last_generation_reached = 0

    try:
        for gen in range(1, cfg.generations + 1):
            collect_metrics = (gen % cfg.snapshot_every == 0 or gen == cfg.generations)
            state, gm, next_id, births = one_generation(
                state, pool, env, rng, cfg, next_id, collect_metrics=collect_metrics
            )
            last_generation_reached = gen
            analysis_births += births
            shadow.reproduce_to_size(state.gids.size)
            max_pop = max(max_pop, state.gids.size)
            if state.gids.size == 0:
                extinct = True
                break
            if gen % cfg.validation_every == 0:
                validate_state(state, pool, cfg)

            # Evaluate MODES snapshots exactly t generations after observation.
            modes.evaluate_due(gen, state)

            if gen % cfg.snapshot_every == 0 or gen == cfg.generations:
                snapshot_index += 1
                comps = real_components(state, pool)
                shadow_comps = shadow.components()
                ar = activity.observe(snapshot_index, gen, comps, shadow_comps, analysis_births)
                ms = modes_snapshot(snapshot_index, gen, analysis_births, state, pool)
                modes.add_snapshot(ms)
                component_snapshots.append({
                    "snapshot": snapshot_index, "generation": gen,
                    "analysis_births": analysis_births, "components": sorted(comps),
                })
                # A reaction is probed once only after it is observed at two consecutive
                # snapshots.  This is a persistence prerequisite for the causal assay,
                # not an OEE score.  The ancestral reaction is not counted as novelty.
                candidates = sorted((comps & prev_components) - ancestral_components - assayed_components)
                for comp in candidates:
                    cr = causal_component_probe(comp, gen, state, pool, env, cfg, task.seed)
                    cr.update({"snapshot": snapshot_index, "analysis_births": analysis_births})
                    causal_raw.append(cr)
                    assayed_components.add(comp)
                prev_components = set(comps)

                unique = np.unique(state.gids)
                max_modules_now = max((pool.total_function_len(int(g)) for g in unique), default=0)
                max_modules = max(max_modules, max_modules_now)
                max_function_diversity = max(max_function_diversity, len(comps))
                max_functional_genotype_diversity = max(max_functional_genotype_diversity, len(unique))
                trajectory.append({
                    "snapshot": snapshot_index, "generation": gen, "analysis_births": analysis_births,
                    **gm,
                    "population": int(state.gids.size),
                    "component_diversity": len(comps),
                    "genotype_pool_size": len(pool.keys),
                    "max_token_id_ever": int(pool.max_token),
                    "max_functions_per_genome": int(max_modules_now),
                })
                # Channon Step-2 shadow reset after every snapshot.
                shadow.reset_from_real(state, pool)

        # Pending MODES rows lacking t future generations are deliberately not evaluated,
        # exactly as the trailing-snapshot exclusion in the standard implementation.
    except RuntimeError as e:
        if str(e).startswith("SAFETY_ABORT"):
            safety_stop = str(e)
        else:
            raise

    runtime = time.time() - t0
    reached_horizon = bool((not extinct) and (not safety_stop) and last_generation_reached >= cfg.generations)
    termination_reason = "horizon" if reached_horizon else ("extinction" if extinct else ("safety_stop" if safety_stop else "terminated"))
    meta = {
        "program_version": PROGRAM_VERSION,
        "run_id": task.run_id, "seed": int(task.seed), "completed": int(reached_horizon),
        "termination_reason": termination_reason,
        "extinct": int(extinct), "safety_stop": safety_stop,
        "generations_requested": cfg.generations,
        "generations_completed": int(last_generation_reached),
        "nmax": cfg.nmax, "n0": cfg.n0, "physical_scale": float(task.physical_scale),
        "scaling": int(task.scaling), "analysis_births": int(analysis_births),
        "runtime_seconds": runtime,
        "max_population": int(max_pop),
        "max_functions_per_genome": int(max_modules),
        "max_population_function_diversity_raw": int(max_function_diversity),
        "max_functional_genotype_diversity_raw": int(max_functional_genotype_diversity),
        "max_active_substrates": int(env.max_active_substrates),
        "max_token_id": int(pool.max_token),
        "genotype_pool_size": len(pool.keys),
        "source_per_capacity": cfg.source_per_capacity,
        "total_external_input": env.total_external_input,
        "causal_probe_repeats": int(cfg.causal_probe_repeats),
        "causal_probe_alpha": float(cfg.causal_probe_alpha),
    }
    if trajectory:
        last = trajectory[-1]
        for k in (
            "pair_birth_fraction", "autonomous_cell_fraction", "genetic_isolated_replacement",
            "genetic_obligate_complementation_fraction", "mean_phi", "mean_alpha",
            "active_substrates", "constructed_mass", "reaction_flux",
            "constructed_resource_flux_fraction", "functional_resource_diversity",
            "local_product_flux_fraction", "local_product_function_diversity",
            "local_metabolic_coordinate_diversity", "local_cascade_energy",
            "local_cascade_carrier_fraction",
            "reaction_function_dependent_cell_fraction", "reaction_metabolic_gain",
            "reaction_energy_rescued_cell_fraction",
        ):
            meta[f"final_{k}"] = last.get(k)

    return {
        "meta": meta,
        "trajectory": trajectory,
        "channon_raw": activity._raw_rows,
        "channon_norm_present": activity._norm_present,
        "min_normalized_activity": activity.minimum_normalized_activity(),
        "modes": modes.rows,
        "causal_raw": causal_raw,
        "component_snapshots": component_snapshots,
    }


# =============================================================================
# Aggregation: Channon Steps 1-3 and MODES, without bespoke OEE gates
# =============================================================================

def _bundle_reached_requested_horizon(bundle: Mapping[str, Any]) -> bool:
    """Full-horizon predicate used only where the published Step-4 scale test needs it.

    Primary Channon Steps 1-3 do NOT use this predicate: as in 11_STANDARD, every
    run contributes every benchmark snapshot it actually reached, including runs
    that later became extinct.
    """
    m = bundle.get("meta", {})
    return bool(
        int(m.get("completed", 0)) == 1
        and int(m.get("extinct", 0)) == 0
        and not str(m.get("safety_stop", "") or "")
        and int(m.get("generations_completed", 0)) >= int(m.get("generations_requested", 0))
    )

def _attach_meta(rows: List[Dict[str, Any]], meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        z = dict(r)
        for k in ("run_id", "seed", "nmax", "physical_scale", "scaling"):
            z[k] = meta.get(k)
        out.append(z)
    return out


def aggregate_primary(bundles: Sequence[Mapping[str, Any]], out_dir: Path) -> Dict[str, Any]:
    # Exact 11_STANDARD primary-campaign policy: every run contributes all snapshots
    # actually observed before its termination.  No full-horizon/extinction filter is
    # added to Steps 1-3.
    primary_bundles = list(bundles)
    complete_count = sum(1 for b in bundles if _bundle_reached_requested_horizon(b))
    global_min = min((float(b.get("min_normalized_activity", 0.0)) for b in primary_bundles), default=0.0)
    threshold = abs(global_min)
    status_rows = [dict(b["meta"]) for b in bundles]
    ch_rows: List[Dict[str, Any]] = []
    modes_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    for b in primary_bundles:
        act = ChannonActivity()
        act._raw_rows = [dict(r) for r in b.get("channon_raw", [])]
        act._norm_present = [{str(k): float(v) for k, v in row.items()} for row in b.get("channon_norm_present", [])]
        ch_rows.extend(_attach_meta(act.finalize(threshold), b["meta"]))
        modes_rows.extend(_attach_meta([dict(r) for r in b.get("modes", [])], b["meta"]))
        trajectory_rows.extend(_attach_meta([dict(r) for r in b.get("trajectory", [])], b["meta"]))
    write_csv(out_dir / "01_RUN_STATUS.csv", status_rows)
    write_csv(out_dir / "02_CHANNON_ACTIVITY_TIMESERIES.csv", ch_rows)
    write_csv(out_dir / "05_MODES_METRICS.csv", modes_rows)
    write_csv(out_dir / "10_MECHANISM_DIAGNOSTICS.csv", trajectory_rows)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in ch_rows:
        by_run[str(r["run_id"])].append(r)
    bounded_rows: List[Dict[str, Any]] = []
    step3_rows: List[Dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        rows.sort(key=lambda r: int(r["snapshot"]))
        x = [float(r["snapshot"]) for r in rows]
        fits = {}
        for metric in ("A_cum", "AN_cum", "AN_median_cum"):
            fit = fit_published_boundedness(x, [float(r[metric]) for r in rows])
            fits[metric] = fit
            bounded_rows.append({
                "run_id": run_id, "metric": metric, "valid": int(bool(fit.get("valid"))),
                "best_model": fit.get("best_model"), "best_family": fit.get("best_family"),
                "best_bic": fit.get("best_bic"),
                "delta_bic_bounded_minus_unbounded": fit.get("delta_bic_bounded_minus_unbounded"),
                "best_params_json": json.dumps(json_safe(fit.get("best_params", {})), sort_keys=True),
            })
        ongoing = bool(rows and float(rows[-1].get("AN_new_running_mean", 0.0)) > 0.0)
        norm_total = bool(fits["AN_cum"].get("valid") and fits["AN_cum"].get("best_family") == "unbounded")
        norm_median = bool(fits["AN_median_cum"].get("valid") and fits["AN_median_cum"].get("best_family") == "unbounded")
        step1 = bool(fits["A_cum"].get("valid") and fits["A_cum"].get("best_family") == "unbounded")
        last = rows[-1] if rows else {}
        step3_rows.append({
            "run_id": run_id,
            "step1_A_cum_unbounded": int(step1),
            "step2_component_normalization_computed": 1,
            "adaptive_threshold_abs_most_negative_aN": threshold,
            "AN_new_time_average_positive_over_observed_horizon": int(ongoing),
            "final_AN_new_running_mean": last.get("AN_new_running_mean", 0.0),
            "AN_cum_unbounded": int(norm_total), "AN_median_cum_unbounded": int(norm_median),
            "step3_observed_horizon_candidate_pattern": int(ongoing and norm_total and norm_median),
            "formal_scope": "Channon long-term pattern; finite horizon reported without mathematical-infinity claim",
            "final_component_diversity": last.get("component_diversity_D_R", 0),
        })
    write_csv(out_dir / "03_CHANNON_BOUNDEDNESS.csv", bounded_rows)
    write_csv(out_dir / "04_CHANNON_STEP3_CRITERIA.csv", step3_rows)

    # Campaign trajectory = replicate mean at each common snapshot, matching the prior standard benchmark.
    by_snap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in ch_rows:
        by_snap[int(r["snapshot"])].append(r)
    campaign_rows = []
    for s in sorted(by_snap):
        rr = by_snap[s]
        campaign_rows.append({
            "snapshot": s, "n_runs": len(rr),
            "A_cum_mean": float(np.mean([float(x["A_cum"]) for x in rr])),
            "AN_cum_mean": float(np.mean([float(x["AN_cum"]) for x in rr])),
            "AN_median_cum_mean": float(np.mean([float(x["AN_median_cum"]) for x in rr])),
            "AN_new_mean": float(np.mean([float(x["AN_new"]) for x in rr])),
        })
    write_csv(out_dir / "04B_CHANNON_CAMPAIGN_TRAJECTORY.csv", campaign_rows)
    cx = [float(r["snapshot"]) for r in campaign_rows]
    cfits = {
        m: fit_published_boundedness(cx, [float(r[m]) for r in campaign_rows])
        for m in ("A_cum_mean", "AN_cum_mean", "AN_median_cum_mean")
    } if campaign_rows else {}
    annew_mean = float(np.mean([r["AN_new_mean"] for r in campaign_rows])) if campaign_rows else 0.0
    campaign_class = {
        "n_runs": len(primary_bundles), "adaptive_threshold_abs_most_negative_aN": threshold,
        "step1_A_cum_unbounded": int(bool(cfits.get("A_cum_mean", {}).get("best_family") == "unbounded")),
        "AN_new_time_average_positive_over_observed_horizon": int(annew_mean > 0.0),
        "AN_new_time_average": annew_mean,
        "AN_cum_unbounded": int(bool(cfits.get("AN_cum_mean", {}).get("best_family") == "unbounded")),
        "AN_median_cum_unbounded": int(bool(cfits.get("AN_median_cum_mean", {}).get("best_family") == "unbounded")),
        "step3_observed_horizon_candidate_pattern": int(
            annew_mean > 0.0
            and cfits.get("AN_cum_mean", {}).get("best_family") == "unbounded"
            and cfits.get("AN_median_cum_mean", {}).get("best_family") == "unbounded"
        ),
        "formal_scope": "finite-horizon empirical classification; mathematical infinity not claimed",
    }
    write_csv(out_dir / "04C_CHANNON_CAMPAIGN_CLASSIFICATION.csv", [campaign_class])

    # MODES four published dimensions, independent rather than collapsed into a score.
    modes_assess: List[Dict[str, Any]] = []
    by_m: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in modes_rows:
        by_m[str(r["run_id"])].append(r)
    for rid, rows in sorted(by_m.items()):
        rows.sort(key=lambda r: int(r["snapshot"]))
        if not rows:
            continue
        x = [float(r["snapshot"]) for r in rows]
        for metric in ("complexity", "ecology"):
            fit = fit_published_boundedness(x, [float(r[metric]) for r in rows])
            modes_assess.append({
                "run_id": rid, "metric": metric, "assessment": "boundedness",
                "valid": int(bool(fit.get("valid"))), "best_model": fit.get("best_model"),
                "best_family": fit.get("best_family"), "best_bic": fit.get("best_bic"),
                "delta_bic_bounded_minus_unbounded": fit.get("delta_bic_bounded_minus_unbounded"),
            })
        modes_assess.append({
            "run_id": rid, "metric": "change", "assessment": "long_run_time_average", "valid": 1,
            "best_family": "positive" if float(rows[-1]["change_running_mean"]) > 0 else "zero",
            "value": float(rows[-1]["change_running_mean"]),
        })
        modes_assess.append({
            "run_id": rid, "metric": "novelty", "assessment": "long_run_time_average", "valid": 1,
            "best_family": "positive" if float(rows[-1]["novelty_running_mean"]) > 0 else "zero",
            "value": float(rows[-1]["novelty_running_mean"]),
        })
    write_csv(out_dir / "06_MODES_ASSESSMENT.csv", modes_assess)

    return {
        "n_runs": len(bundles), "n_complete_runs": complete_count,
        "campaign_adaptive_threshold": threshold,
        "campaign_step3_observed_horizon_candidate": bool(campaign_class["step3_observed_horizon_candidate_pattern"]),
        "formal_scope": "published Channon/MODES measurements only; no bespoke OEE score",
    }


# =============================================================================
# Step 4/5 physical scalability
# =============================================================================

def aggregate_scaling(bundles: Sequence[Mapping[str, Any]], out_dir: Path, capacities: Sequence[int], expected_runs: int) -> Dict[str, Any]:
    complete = [b for b in bundles if _bundle_reached_requested_horizon(b)]
    global_min = min((float(b.get("min_normalized_activity", 0.0)) for b in complete), default=0.0)
    threshold = abs(global_min)
    run_rows = []
    for b in complete:
        act = ChannonActivity(); act._raw_rows = [dict(r) for r in b.get("channon_raw", [])]
        act._norm_present = [{str(k): float(v) for k, v in row.items()} for row in b.get("channon_norm_present", [])]
        rows = act.finalize(threshold)
        m = dict(b["meta"])
        m.update({
            "adaptive_threshold_abs_most_negative_aN": threshold,
            "max_adaptive_component_diversity": max((int(r["adaptive_component_diversity"]) for r in rows), default=0),
            "max_component_diversity_D_R": max((int(r["component_diversity_D_R"]) for r in rows), default=0),
        })
        run_rows.append(m)
    write_csv(out_dir / "07_STEP4_SCALABILITY_RUNS.csv", run_rows)

    bycap: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in run_rows:
        bycap[int(r["nmax"])].append(r)
    summary = []
    for cap in capacities:
        rr = bycap.get(int(cap), [])
        row: Dict[str, Any] = {
            "nmax": int(cap), "physical_scale_factor": int(cap) / float(BASE_CAPACITY),
            "expected_runs": expected_runs, "completed_runs": len(rr),
            "eligible_for_upper_bound_sequence": int(len(rr) == expected_runs),
            "adaptive_threshold_abs_most_negative_aN": threshold,
        }
        if rr:
            row.update({
                "upper_bound_max_adaptive_component_diversity": max(int(x["max_adaptive_component_diversity"]) for x in rr),
                "median_run_max_adaptive_component_diversity": float(np.median([int(x["max_adaptive_component_diversity"]) for x in rr])),
                "upper_bound_max_component_diversity_D_R": max(int(x["max_component_diversity_D_R"]) for x in rr),
                "upper_bound_max_functions_per_genome": max(int(x["max_functions_per_genome"]) for x in rr),
                "upper_bound_max_population_function_diversity_raw": max(int(x["max_population_function_diversity_raw"]) for x in rr),
                "upper_bound_max_active_substrates": max(int(x["max_active_substrates"]) for x in rr),
                "source_per_capacity_mean": float(np.mean([float(x["source_per_capacity"]) for x in rr])),
            })
        summary.append(row)
    write_csv(out_dir / "08_STEP4_SCALABILITY_SUMMARY.csv", summary)

    eligible = [r for r in summary if int(r.get("eligible_for_upper_bound_sequence", 0)) == 1 and "upper_bound_max_adaptive_component_diversity" in r]
    eligible.sort(key=lambda r: int(r["nmax"]))
    record_sequence = []
    record = -math.inf
    for r in eligible:
        v = float(r["upper_bound_max_adaptive_component_diversity"])
        if v > record:
            record_sequence.append({"nmax": int(r["nmax"]), "physical_scale_factor": float(r["physical_scale_factor"]), "upper_bound": int(v)})
            record = v

    # Step 5 descriptive order, no pass/fail threshold.
    step5 = []
    if len(record_sequence) >= 3:
        x = np.asarray([float(r["nmax"]) for r in eligible])
        y = np.asarray([float(r["upper_bound_max_adaptive_component_diversity"]) for r in eligible])
        models = []
        X = np.column_stack([np.ones(len(x)), np.log2(x / BASE_CAPACITY)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ beta
        models.append(("logarithmic", float(np.sum((y-pred)**2)), {"intercept_at_base": float(beta[0]), "slope_per_doubling": float(beta[1])}))
        X = np.column_stack([np.ones(len(x)), x / BASE_CAPACITY])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ beta
        models.append(("linear", float(np.sum((y-pred)**2)), {"intercept": float(beta[0]), "slope_per_base_capacity": float(beta[1])}))
        if np.all(y > 0):
            lx = np.log(x / BASE_CAPACITY); ly = np.log(y)
            X = np.column_stack([np.ones(len(x)), lx]); beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
            a, b = math.exp(float(beta[0])), float(beta[1]); pred = a * np.power(x / BASE_CAPACITY, b)
            models.append(("power", float(np.sum((y-pred)**2)), {"coefficient_at_base": a, "exponent": b}))
        scored = sorted([(name, _bic(rss, len(y), 2), rss, params) for name, rss, params in models], key=lambda z: (z[1], z[2]))
        step5.append({
            "metric": "upper_bound_max_adaptive_component_diversity", "n_scale_points": len(x),
            "best_descriptive_order": scored[0][0], "best_bic": scored[0][1],
            "best_params_json": json.dumps(json_safe(scored[0][3]), sort_keys=True),
            "all_models_json": json.dumps([{"model":z[0],"bic":z[1],"rss":z[2],"params":z[3]} for z in scored], sort_keys=True),
            "formal_scope": "descriptive order over tested physical scales; no Step-5 pass/fail threshold",
        })
    write_csv(out_dir / "10_STEP5_SCALING_ORDER.csv", step5)
    return {
        "scaling_adaptive_threshold": threshold,
        "primary_step4_quantity": "system-level adaptive component diversity from Channon component-normalized activity",
        "adaptive_diversity_upper_bounds": [int(r["upper_bound_max_adaptive_component_diversity"]) for r in eligible],
        "step4_greater_upper_bound_sequence": record_sequence,
        "step4_extent_observed": max(0, len(record_sequence)-1),
        "step5_status": "characterized" if step5 else "not_applicable_without_at_least_three_greater_upper_bound_points",
    }


# =============================================================================
# Driver / output
# =============================================================================

def configure(args: argparse.Namespace, capacity: Optional[int] = None) -> ModelConfig:
    if args.mode == "smoke":
        cap = int(capacity or min(BASE_CAPACITY, 96))
        return ModelConfig(
            n0=cap, nmax=cap, generations=min(args.generations or 2_000, 2_000),
            snapshot_every=50, validation_every=250, modes_filter_generations=16,
            causal_probe_repeats=3,
        )
    cap = int(capacity or BASE_CAPACITY)
    g = int(args.generations or 40_000)
    return ModelConfig(n0=cap, nmax=cap, generations=g)


def build_tasks(args: argparse.Namespace, scaling: bool = False, capacity: Optional[int] = None) -> List[RunTask]:
    cfg = configure(args, capacity)
    nruns = (1 if args.mode == "smoke" else (args.scale_runs if scaling else args.runs))
    tasks = []
    for i in range(int(nruns)):
        seed = stable_seed(args.root_seed, "scale" if scaling else "primary", capacity or cfg.nmax, i)
        rid = (f"scale{cfg.nmax}_run{i:03d}" if scaling else f"run_{i:03d}")
        tasks.append(RunTask(rid, seed, cfg, cfg.nmax / float(BASE_CAPACITY), scaling))
    return tasks


def run_tasks(tasks: Sequence[RunTask], workers: int, out_dir: Path, subdir: str) -> List[Dict[str, Any]]:
    d = out_dir / subdir; d.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    pending: List[RunTask] = []
    for t in tasks:
        p = d / f"{t.run_id}.json.gz"
        if p.exists():
            try:
                cached = read_json_gz(p)
                if str(cached.get("meta", {}).get("program_version", "")) == PROGRAM_VERSION:
                    results.append(cached); continue
            except Exception:
                pass
        pending.append(t)
    if not pending:
        return results
    if workers <= 1:
        for i, t in enumerate(pending, 1):
            b = simulate_task(t); write_json_gz(d / f"{t.run_id}.json.gz", b); results.append(b)
            status = "complete" if int(b["meta"].get("completed", 0)) else str(b["meta"].get("termination_reason", "terminated"))
            print(f"[{i}/{len(pending)}] {t.run_id} {status}; runtime={b['meta']['runtime_seconds']:.2f}s", flush=True)
    else:
        # Run at most one scientific run per worker process, then recycle the pool.
        # Long evolutionary runs allocate many short-lived genotype and SciPy/NumPy
        # objects; Python allocators may retain that memory after a task finishes.
        # Reusing the same worker for a second run caused severe second-batch slowdown
        # on desktop-class systems.  Fresh-process batches keep throughput stable and
        # do not change RNG streams or scientific state.
        ctx = mp.get_context("spawn")
        done = 0
        for start in range(0, len(pending), workers):
            batch = pending[start:start + workers]
            with ProcessPoolExecutor(max_workers=min(workers, len(batch)), mp_context=ctx) as ex:
                futs = {ex.submit(simulate_task, t): t for t in batch}
                for fut in as_completed(futs):
                    t = futs[fut]; b = fut.result(); done += 1
                    write_json_gz(d / f"{t.run_id}.json.gz", b); results.append(b)
                    status = "complete" if int(b["meta"].get("completed", 0)) else str(b["meta"].get("termination_reason", "terminated"))
                    print(f"[{done}/{len(pending)}] {t.run_id} {status}; runtime={b['meta']['runtime_seconds']:.2f}s", flush=True)
    return results


def write_report(
    out: Path, primary: Mapping[str, Any], hallmarks: Mapping[str, Any],
    scaling: Optional[Mapping[str, Any]], args: argparse.Namespace,
) -> None:
    lines = [
        "# Reproductive Interdependence — Multi-Hallmark OEE Evaluation", "",
        f"Program version: {PROGRAM_VERSION}", "",
        "## Evaluation policy", "",
        "No single metric is treated as the definition of OEE and no composite OEE score is calculated.",
        "Primary behavioral evidence is direct causal adaptive novelty plus its ongoing accumulation and adaptive-component diversity growth.",
        "MODES and Channon activity statistics are reported independently; disagreement among measures is retained rather than resolved by an extra gate.", "",
        "## Direct adaptive-novelty assay", "",
        "A newly persistent hereditary reaction is tested by paired live-state cloning. The knockout removes only that exact reaction transformation from current carriers.",
        "Mutation is disabled during the one-generation probe; ecology, pair state, basal investment and interaction traits are unchanged.",
        "Tagged carrier-descendant abundance is compared over paired repeats. Full mode uses 8 repeats. Positive causal effect, nominal p<=0.05 and within-run BH q<=0.05 are reported as separate evidence tiers; none is used as a universal OEE gate.", "",
        "## Behavioral-hallmark summary", "", "```json", json.dumps(json_safe(hallmarks), indent=2), "```", "",
        "## Independent Channon/MODES reference summary", "", "```json", json.dumps(json_safe(primary), indent=2), "```", "",
        "## Interpretation", "",
        "- Adaptive novelty: 07_CAUSAL_ADAPTIVE_NOVELTY_ASSAYS.csv",
        "- Ongoing adaptive novelty and adaptive diversity: 08_BEHAVIORAL_HALLMARK_TIMESERIES.csv and 09_OEE_HALLMARK_SUMMARY.csv",
        "- Ecological expansion: 09B_ECOLOGICAL_EXPANSION_SUMMARY.csv",
        "- MODES: 05_MODES_METRICS.csv and 06_MODES_ASSESSMENT.csv",
        "- Channon reference: 02-04C files.",
        "Finite horizons provide empirical evidence only; mathematical infinity is not claimed.", "",
    ]
    if scaling is not None:
        lines += ["## Physical-scale evidence", "", "```json", json.dumps(json_safe(scaling), indent=2), "```", ""]
    lines += [
        "## Literature framing", "",
        "The 2024 OEE special-issue editorial emphasizes behavioral hallmarks—ongoing adaptive novelty and ongoing complexity growth—while noting less agreement about mechanisms.",
        "MODES treats change, novelty, complexity and ecology as separate measurements. Channon (2024) is retained here as a specific Tokyo-Type-1 testing procedure, not as the sole definition of OEE.",
    ]
    atomic_text(out / "10_REPORT.md", "\n".join(lines) + "\n")


# =============================================================================
# Structural / numerical self-tests
# =============================================================================

def _load_reference_11() -> Optional[Any]:
    candidates = [
        Path(__file__).with_name("11_STANDARD_OEE_BENCHMARK_VALIDATION.py"),
        Path("/mnt/data/oee_reproductive/11_STANDARD_OEE_BENCHMARK_VALIDATION.py"),
        Path.home() / "Downloads" / "11_STANDARD_OEE_BENCHMARK_VALIDATION.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("reference_11", p)
            mod = importlib.util.module_from_spec(spec); assert spec and spec.loader
            spec.loader.exec_module(mod)
            return mod
    return None


def self_test() -> None:
    print("12 self-test starting", flush=True)
    cfg = ModelConfig(n0=48, nmax=48, generations=20, snapshot_every=5, validation_every=5, modes_filter_generations=4)
    pool = GenotypePool(cfg.basal_functions)
    state, nid = initial_state(cfg, pool)
    validate_state(state, pool, cfg)
    # Mutation can create variable-length genomes without a truncation cap.
    rng = np.random.default_rng(123)
    key = pool.key(0)
    seen_len = {len(key[1])}
    for _ in range(20_000):
        if rng.random() < 0.3:
            key = mutate_genome_key(key, rng, replace(cfg, mu_module_dup_per_module=0.02, mu_module_add=0.02))
            if len(key[1]) > 1000:
                break
            seen_len.add(len(key[1]))
    assert max(seen_len) > 1
    print("PASS variable-length hereditary reaction genome", flush=True)

    # Reaction layer conserves substrate mass apart from harvest, source, and
    # dissipation.  The ancestral 0->1 reaction has no exact local downstream
    # consumer, so retained product must be released rather than silently lost.
    env = ReactionEnvironment(cfg)
    before = float(np.sum(env.mass))
    energy, support, em = env.step(state, pool, cfg)
    after = float(np.sum(env.mass))
    expected_after_pre_reaction = before * (1-cfg.substrate_dissipation) + cfg.source_per_capacity*cfg.nmax
    assert abs((after + float(em["harvest"])) - expected_after_pre_reaction) < 1e-7
    assert np.all(energy >= 0)
    assert support.shape == (state.gids.size, cfg.basal_functions)
    assert not np.any(support), "V11 must not project token identity onto the finite basal vector"

    # Exact-token coordinate test.  Both genotypes have the same number of modules
    # and the same primary 0->X reaction opportunity.  Only ((0,1),(1,2)) has exact
    # local connectivity: retained token 1 can feed 1->2.  In ((0,3),(1,2)), token 3
    # is a distinct coordinate and cannot be substituted for token 1.  No novelty or
    # token-number reward is involved.
    chain_gid = pool.intern((pool.key(0)[0], ((0, 1), (1, 2))))
    mismatch_gid = pool.intern((pool.key(0)[0], ((0, 3), (1, 2))))
    def mk_state(gid: int) -> PopulationState:
        nn = 24
        return PopulationState(
            gids=np.full(nn, gid, dtype=np.int32),
            x=np.full((nn, cfg.basal_functions), cfg.initial_x, dtype=np.float32),
            phi=np.zeros(nn, dtype=np.float32), alpha=np.zeros(nn, dtype=np.float32),
            partner=np.full(nn, -1, dtype=np.int32),
            individual_id=np.arange(1, nn+1, dtype=np.int64),
            ancestors=np.zeros((nn, cfg.modes_filter_generations), dtype=np.int64),
        )
    c_cfg = replace(cfg, n0=24, nmax=24)
    cenv = ReactionEnvironment(c_cfg)
    menv = ReactionEnvironment(c_cfg)
    ce, cs, cem = cenv.step(mk_state(chain_gid), pool, c_cfg)
    me, ms, mem = menv.step(mk_state(mismatch_gid), pool, c_cfg)
    assert not np.any(cs) and not np.any(ms)
    assert cem["local_product_flux"] > mem["local_product_flux"] + 1e-12
    assert cem["local_metabolic_coordinate_diversity"] >= 1
    assert float(np.mean(ce)) > float(np.mean(me)) + 1e-12
    print("PASS exact-token metabolic coordinates and dissipative mass accounting", flush=True)

    # Dynamics / partner integrity / ancestry ring.
    for _ in range(20):
        state, gm, nid, births = one_generation(state, pool, env, rng, cfg, nid)
        if state.gids.size == 0:
            break
        validate_state(state, pool, cfg)
    print("PASS population dynamics and reciprocal pair integrity", flush=True)

    # Channon numerical parity against the locked previous standard implementation when available.
    ref = _load_reference_11()
    if ref is not None:
        a = ChannonActivity(); b = ref.ChannonActivity()
        obs = [
            ({"A","B"},{"A"}), ({"A","B","C"},{"B","D"}),
            ({"A","C"},{"A","D"}), ({"A","C","E"},{"C"}),
            ({"A","E","F"},{"A","F"}),
        ]
        for i,(r,s) in enumerate(obs,1):
            a.observe(i,i*10,set(r),set(s),i*100); b.observe(i,i*10,set(r),set(s),i*100)
        th = max(abs(a.minimum_normalized_activity()), abs(b.minimum_normalized_activity()))
        ra = a.finalize(th); rb = b.finalize(th)
        for x,y in zip(ra,rb):
            for k in ("A_cum","AN_cum","AN_median_cum","AN_new","AN_new_running_mean"):
                assert abs(float(x[k])-float(y[k])) < 1e-12, (k,x[k],y[k])
        x = np.arange(1,10,dtype=float); y = np.power(x,0.4)
        fa = fit_published_boundedness(x,y); fb = ref.fit_published_boundedness(x,y)
        assert fa.get("best_family") == fb.get("best_family")
        print("PASS numerical parity with 11_STANDARD Channon and boundedness logic", flush=True)
    else:
        print("SKIP reference-11 parity test: file not found", flush=True)

    # Short complete benchmark run.
    t = RunTask("selftest", 999, cfg)
    b = simulate_task(t)
    assert b["meta"]["generations_completed"] > 0
    assert len(b["channon_raw"]) > 0
    assert "causal_raw" in b and "component_snapshots" in b
    # Direct causal probe returns a finite paired intervention result without altering
    # the live model state.
    cp = GenotypePool(cfg.basal_functions)
    mask=(1<<cfg.basal_functions)-1
    gid=cp.intern((mask, ((0,1),(1,2))))
    ps=PopulationState(
        np.full(32,gid,np.int32), np.full((32,cfg.basal_functions),cfg.initial_x,np.float32),
        np.zeros(32,np.float32), np.zeros(32,np.float32), np.full(32,-1,np.int32),
        np.arange(1,33,dtype=np.int64), np.zeros((32,cfg.modes_filter_generations),np.int64))
    pe=ReactionEnvironment(replace(cfg,n0=32,nmax=32,causal_probe_repeats=3)); pe.ensure(2); pe.mass[1]=200.0
    pr=causal_component_probe("R1>2",5,ps,cp,pe,replace(cfg,n0=32,nmax=32,causal_probe_repeats=3),1234)
    assert np.isfinite(float(pr["mean_log_descendant_effect"])) and 0.0 <= float(pr["p_value"]) <= 1.0
    print("PASS direct paired causal reaction knockout assay", flush=True)
    print("PASS end-to-end short benchmark run", flush=True)

    # Primary aggregation parity policy: an extinct run must still contribute every
    # Channon snapshot it reached, exactly as in 11_STANDARD.  This guards against
    # reintroducing the stricter V10 full-horizon filter.
    b2 = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in b.items()}
    b2["meta"] = dict(b["meta"])
    b2["meta"]["run_id"] = "selftest_extinct"
    b2["meta"]["completed"] = 0
    b2["meta"]["extinct"] = 1
    # Keep only the first observed half to mimic a trajectory that terminates early.
    for k in ("channon_raw", "channon_norm_present", "modes", "trajectory"):
        vals = list(b.get(k, []))
        b2[k] = vals[:max(1, len(vals)//2)] if vals else []
    td = Path(tempfile.mkdtemp(prefix="oee11_selftest_"))
    try:
        aggregate_primary([b, b2], td)
        with (td / "02_CHANNON_ACTIVITY_TIMESERIES.csv").open("r", newline="", encoding="utf-8") as fh:
            ids = {str(r["run_id"]) for r in csv.DictReader(fh)}
        assert "selftest" in ids and "selftest_extinct" in ids
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("PASS 11_STANDARD partial-trajectory aggregation policy", flush=True)
    print("ALL 12 STRUCTURAL SELF-TESTS PASSED", flush=True)


# =============================================================================
# npj Complexity mechanism-validation extension
# =============================================================================

MECHANISM_CONDITIONS: Tuple[str, ...] = ("FULL", "INT_MINUS", "ECO_MINUS", "BOTH_MINUS")


def _condition_flags(condition: str) -> Tuple[bool, bool]:
    c = str(condition).upper()
    if c == "FULL":
        return True, True
    if c == "INT_MINUS":
        return False, True
    if c == "ECO_MINUS":
        return True, False
    if c == "BOTH_MINUS":
        return False, False
    raise ValueError(f"unknown mechanism condition: {condition}")


def configure_mechanism(args: argparse.Namespace, condition: str) -> ModelConfig:
    interdependence, ecology = _condition_flags(condition)
    if args.mode == "smoke":
        g = int(args.generations or 400)
        return ModelConfig(
            n0=BASE_CAPACITY, nmax=BASE_CAPACITY, generations=g,
            snapshot_every=max(20, min(50, g // 4 if g >= 80 else 20)),
            validation_every=max(50, min(250, g)), modes_filter_generations=16,
            causal_probe_repeats=3,
            mechanism_condition=str(condition),
            functional_interdependence_enabled=interdependence,
            shared_ecological_construction_enabled=ecology,
        )
    g = int(args.generations or 40_000)
    return ModelConfig(
        n0=BASE_CAPACITY, nmax=BASE_CAPACITY, generations=g,
        mechanism_condition=str(condition),
        functional_interdependence_enabled=interdependence,
        shared_ecological_construction_enabled=ecology,
    )


def build_mechanism_tasks(args: argparse.Namespace) -> List[RunTask]:
    requested = [x.strip().upper() for x in str(args.conditions).split(",") if x.strip()]
    unknown = [x for x in requested if x not in MECHANISM_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}; allowed={MECHANISM_CONDITIONS}")
    nruns = 1 if args.mode == "smoke" else int(args.runs)
    tasks: List[RunTask] = []
    # Matched replicate identity: all four conditions receive the same seed for a
    # replicate. Divergence therefore reflects the intervention rather than seed choice.
    for i in range(nruns):
        seed = stable_seed(args.root_seed, "mechanism_matched", i)
        for condition in requested:
            cfg = configure_mechanism(args, condition)
            rid = f"{condition}_run_{i:03d}"
            tasks.append(RunTask(rid, seed, cfg, 1.0, False))
    return tasks


def _carrier_mask_for_component(component: str, state: PopulationState, pool: GenotypePool) -> np.ndarray:
    target = _component_tuple(component)
    carrier = np.zeros(state.gids.size, dtype=bool)
    for gid in np.unique(state.gids):
        if target in pool.key(int(gid))[1]:
            carrier[state.gids == gid] = True
    return carrier


def _component_carrier_gids(state: PopulationState, pool: GenotypePool) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = defaultdict(set)
    for gid in np.unique(state.gids):
        for comp in pool.components(int(gid)):
            out[str(comp)].add(int(gid))
    return out


def _scc_longest_depth(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Tuple[int, int]:
    """Return (number of SCCs, longest edge depth in the SCC condensation DAG)."""
    ns = list(dict.fromkeys(str(x) for x in nodes))
    adj: Dict[str, List[str]] = {n: [] for n in ns}
    radj: Dict[str, List[str]] = {n: [] for n in ns}
    for a, b in edges:
        a = str(a); b = str(b)
        if a not in adj:
            adj[a] = []; radj[a] = []; ns.append(a)
        if b not in adj:
            adj[b] = []; radj[b] = []; ns.append(b)
        adj[a].append(b); radj[b].append(a)
    seen: Set[str] = set(); order: List[str] = []
    def dfs1(v: str) -> None:
        seen.add(v)
        for w in adj[v]:
            if w not in seen: dfs1(w)
        order.append(v)
    for n in ns:
        if n not in seen: dfs1(n)
    comp_id: Dict[str, int] = {}
    def dfs2(v: str, cid: int) -> None:
        comp_id[v] = cid
        for w in radj[v]:
            if w not in comp_id: dfs2(w, cid)
    cid = 0
    for n in reversed(order):
        if n not in comp_id:
            dfs2(n, cid); cid += 1
    dag: Dict[int, Set[int]] = {i: set() for i in range(cid)}
    indeg = [0] * cid
    for a, b in edges:
        ca, cb = comp_id[str(a)], comp_id[str(b)]
        if ca != cb and cb not in dag[ca]:
            dag[ca].add(cb); indeg[cb] += 1
    q = deque(i for i in range(cid) if indeg[i] == 0)
    depth = [0] * cid
    while q:
        u = q.popleft()
        for v in dag[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    return cid, max(depth, default=0)


def dependency_network_snapshot(
    snapshot: int, generation: int, state: PopulationState, pool: GenotypePool,
    env: ReactionEnvironment, cfg: ModelConfig, component_first_seen: Mapping[str, int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    carriers = _component_carrier_gids(state, pool)
    components = sorted(carriers)
    realized: Set[str] = set()
    eps = 1e-12
    for comp in components:
        a, b = _component_tuple(comp)
        if a < env.mass.size and float(env.mass[a]) > eps:
            realized.add(comp)
    edge_rows: List[Dict[str, Any]] = []
    edge_pairs: List[Tuple[str, str]] = []
    if cfg.shared_ecological_construction_enabled:
        producers: Dict[int, List[str]] = defaultdict(list)
        consumers: Dict[int, List[str]] = defaultdict(list)
        for comp in realized:
            a, b = _component_tuple(comp)
            if b > 0: producers[b].append(comp)
            if a > 0: consumers[a].append(comp)
        for token in sorted(set(producers) & set(consumers)):
            if token >= env.mass.size or float(env.mass[token]) <= eps:
                continue
            for p in sorted(producers[token]):
                for c in sorted(consumers[token]):
                    if p == c: continue
                    pg = carriers[p]; cg = carriers[c]
                    cross = int(any(x != y for x in pg for y in cg))
                    historical = int(int(component_first_seen.get(p, generation)) < int(component_first_seen.get(c, generation)))
                    edge_pairs.append((p, c))
                    edge_rows.append({
                        "snapshot": int(snapshot), "generation": int(generation),
                        "condition": cfg.mechanism_condition,
                        "producer_component": p, "resource_token": int(token),
                        "consumer_component": c, "resource_mass": float(env.mass[token]),
                        "cross_genotype_possible": cross,
                        "historically_ordered": historical,
                        "producer_first_seen": int(component_first_seen.get(p, generation)),
                        "consumer_first_seen": int(component_first_seen.get(c, generation)),
                    })
    scc_n, depth = _scc_longest_depth(components, edge_pairs)
    return ({
        "snapshot": int(snapshot), "generation": int(generation),
        "condition": cfg.mechanism_condition,
        "n_components": len(components), "n_realized_components": len(realized),
        "n_dependency_edges": len(edge_pairs),
        "n_cross_genotype_edges": sum(int(r["cross_genotype_possible"]) for r in edge_rows),
        "n_historically_ordered_edges": sum(int(r["historically_ordered"]) for r in edge_rows),
        "n_dependency_scc": int(scc_n), "dependency_depth": int(depth),
    }, edge_rows)


def historical_resource_dependency_probe(
    component: str, generation: int, state: PopulationState, pool: GenotypePool,
    env: ReactionEnvironment, cfg: ModelConfig, root_seed: int,
    component_first_seen: Mapping[str, int], token_first_seen: Mapping[int, int],
) -> Dict[str, Any]:
    a, b = _component_tuple(component)
    carrier = _carrier_mask_for_component(component, state, pool)
    carrier_ids = state.individual_id[carrier].copy()
    mass_a = float(env.mass[a]) if a < env.mass.size else 0.0
    upstream_current = sorted({
        f"R{x}>{y}" for gid in np.unique(state.gids) for x, y in pool.key(int(gid))[1]
        if int(y) == int(a) and f"R{x}>{y}" != component
    }) if a > 0 else []
    target_carrier_gids = set(int(x) for x in np.unique(state.gids[carrier])) if np.any(carrier) else set()
    local_upstream_carriers = 0
    if target_carrier_gids:
        local_upstream_carriers = sum(
            int(np.count_nonzero((state.gids == gid) & carrier))
            for gid in target_carrier_gids
            if any(int(y) == int(a) and (int(x), int(y)) != (a, b) for x, y in pool.key(gid)[1])
        )
    local_upstream_fraction = float(local_upstream_carriers / max(1, carrier_ids.size))
    first_upstream = min((int(component_first_seen.get(x, generation)) for x in upstream_current), default=-1)
    target_first = int(component_first_seen.get(component, generation))
    token_first = int(token_first_seen.get(int(a), generation)) if a > 0 else -1
    historical_precedence = int(a > 0 and token_first < target_first)
    strict_shared_candidate = int(a > 0 and mass_a > 1e-12 and local_upstream_carriers == 0 and historical_precedence == 1)
    base = {
        "component": component, "generation": int(generation), "input_token": int(a), "output_token": int(b),
        "n_carriers": int(carrier_ids.size), "input_resource_mass": mass_a,
        "upstream_components_current": ";".join(upstream_current),
        "first_upstream_component_generation": int(first_upstream),
        "target_first_seen_generation": int(target_first), "resource_first_seen_generation": int(token_first),
        "historical_resource_precedes_target": historical_precedence,
        "local_upstream_carrier_fraction": local_upstream_fraction,
        "strict_shared_dependency_candidate": strict_shared_candidate,
        "repeats": int(cfg.causal_probe_repeats),
    }
    if a <= 0 or mass_a <= 1e-12 or carrier_ids.size == 0:
        return {**base, "eligible": 0, "mean_log_erasure_effect": 0.0, "p_erasure": 1.0,
                "mean_log_rescue_recovery": 0.0, "rescue_matches_intact_fraction": 0.0,
                "intact_descendants_json": "[]", "erased_descendants_json": "[]", "rescued_descendants_json": "[]"}
    pcfg = _probe_cfg(cfg)
    assay_state0, assay_pool = _remap_active_state_for_assay(state, pool, pcfg, None)
    intact_counts: List[int] = []; erased_counts: List[int] = []; rescued_counts: List[int] = []
    erasure_effects: List[float] = []; rescue_recovery: List[float] = []
    for r in range(int(cfg.causal_probe_repeats)):
        seed = stable_seed(root_seed, "resource_dependency", generation, component, r)
        next_id = int(max(int(state.individual_id.max(initial=0)) + 1, 1))
        states = [_copy_state(assay_state0) for _ in range(3)]
        envs = [_assay_env_clone(env, pcfg) for _ in range(3)]
        # Intact / erased / rescue. Rescue re-adds exactly the pre-intervention mass,
        # serving as a specificity control for the resource erasure.
        if a >= envs[1].mass.size: envs[1].ensure(a)
        envs[1].mass[a] = 0.0
        if a >= envs[2].mass.size: envs[2].ensure(a)
        envs[2].mass[a] = 0.0; envs[2].mass[a] += mass_a
        outs=[]
        for j in range(3):
            rr = np.random.default_rng(seed)
            child, _, _, _ = one_generation(states[j], assay_pool, envs[j], rr, pcfg, next_id, collect_metrics=False)
            outs.append(_tagged_children(child, carrier_ids))
        ni, ne, nr = outs
        intact_counts.append(ni); erased_counts.append(ne); rescued_counts.append(nr)
        erasure_effects.append(float(math.log1p(ni) - math.log1p(ne)))
        rescue_recovery.append(float(math.log1p(nr) - math.log1p(ne)))
    return {
        **base, "eligible": 1,
        "mean_log_erasure_effect": float(np.mean(erasure_effects)),
        "p_erasure": _exact_signflip_p_greater(erasure_effects),
        "mean_log_rescue_recovery": float(np.mean(rescue_recovery)),
        "rescue_matches_intact_fraction": float(np.mean(np.asarray(rescued_counts) == np.asarray(intact_counts))),
        "intact_descendants_json": json.dumps(intact_counts),
        "erased_descendants_json": json.dumps(erased_counts),
        "rescued_descendants_json": json.dumps(rescued_counts),
    }


def simulate_mechanism_task(task: RunTask) -> Dict[str, Any]:
    t0 = time.time(); cfg = task.cfg
    rng = np.random.default_rng(task.seed)
    pool = GenotypePool(cfg.basal_functions)
    state, next_id = initial_state(cfg, pool)
    env = ReactionEnvironment(cfg)
    shadow = NeutralShadow(cfg, task.seed); shadow.reset_from_real(state, pool)
    activity = ChannonActivity(); modes = OnlineModes(cfg.modes_filter_generations)
    causal_raw: List[Dict[str, Any]] = []; resource_raw: List[Dict[str, Any]] = []
    component_snapshots: List[Dict[str, Any]] = []; dependency_snapshots: List[Dict[str, Any]] = []; dependency_edges: List[Dict[str, Any]] = []
    assayed_components: Set[str] = set(); prev_components: Set[str] = set()
    ancestral_components = real_components(state, pool)
    component_first_seen: Dict[str, int] = {str(c): 0 for c in ancestral_components}
    token_first_seen: Dict[int, int] = {0: 0}
    analysis_births = 0; snapshot_index = 0; trajectory: List[Dict[str, Any]] = []
    max_pop = state.gids.size; max_modules = 1; max_function_diversity = len(ancestral_components); max_functional_genotype_diversity = 1
    extinct = False; safety_stop = ""; last_generation_reached = 0
    try:
        for gen in range(1, cfg.generations + 1):
            collect_metrics = (gen % cfg.snapshot_every == 0 or gen == cfg.generations)
            state, gm, next_id, births = one_generation(state, pool, env, rng, cfg, next_id, collect_metrics=collect_metrics)
            last_generation_reached = gen; analysis_births += births; shadow.reproduce_to_size(state.gids.size)
            max_pop = max(max_pop, state.gids.size)
            if state.gids.size == 0:
                extinct = True; break
            if gen % cfg.validation_every == 0: validate_state(state, pool, cfg)
            modes.evaluate_due(gen, state)
            if collect_metrics:
                snapshot_index += 1
                comps = real_components(state, pool); shadow_comps = shadow.components()
                for c in comps: component_first_seen.setdefault(str(c), int(gen))
                for tok in np.flatnonzero(env.mass > 1e-9): token_first_seen.setdefault(int(tok), int(gen))
                activity.observe(snapshot_index, gen, comps, shadow_comps, analysis_births)
                modes.add_snapshot(modes_snapshot(snapshot_index, gen, analysis_births, state, pool))
                component_snapshots.append({"snapshot": snapshot_index, "generation": gen, "analysis_births": analysis_births, "components": sorted(comps)})
                candidates = sorted((comps & prev_components) - ancestral_components - assayed_components)
                for comp in candidates:
                    cr = causal_component_probe(comp, gen, state, pool, env, cfg, task.seed)
                    cr.update({"snapshot": snapshot_index, "analysis_births": analysis_births, "condition": cfg.mechanism_condition})
                    causal_raw.append(cr); assayed_components.add(comp)
                    if float(cr.get("mean_log_descendant_effect", 0.0)) > 0.0 and _component_tuple(comp)[0] > 0:
                        rr = historical_resource_dependency_probe(comp, gen, state, pool, env, cfg, task.seed, component_first_seen, token_first_seen)
                        rr.update({"snapshot": snapshot_index, "analysis_births": analysis_births, "condition": cfg.mechanism_condition})
                        resource_raw.append(rr)
                prev_components = set(comps)
                dep, edges = dependency_network_snapshot(snapshot_index, gen, state, pool, env, cfg, component_first_seen)
                dep.update({"analysis_births": analysis_births}); dependency_snapshots.append(dep)
                for e in edges:
                    e.update({"run_id": task.run_id, "seed": int(task.seed), "analysis_births": analysis_births})
                dependency_edges.extend(edges)
                unique = np.unique(state.gids)
                max_modules_now = max((pool.total_function_len(int(g)) for g in unique), default=0)
                max_modules = max(max_modules, max_modules_now); max_function_diversity = max(max_function_diversity, len(comps)); max_functional_genotype_diversity = max(max_functional_genotype_diversity, len(unique))
                trajectory.append({
                    "snapshot": snapshot_index, "generation": gen, "analysis_births": analysis_births,
                    **gm, "population": int(state.gids.size), "component_diversity": len(comps),
                    "genotype_pool_size": len(pool.keys), "max_token_id_ever": int(pool.max_token),
                    "max_functions_per_genome": int(max_modules_now),
                    "structural_mutation_events_cumulative": int(pool.structural_mutation_events),
                    "reaction_identities_ever": int(len(pool.component_identities_ever)),
                })
                shadow.reset_from_real(state, pool)
    except RuntimeError as e:
        if str(e).startswith("SAFETY_ABORT"): safety_stop = str(e)
        else: raise
    runtime = time.time() - t0
    reached_horizon = bool((not extinct) and (not safety_stop) and last_generation_reached >= cfg.generations)
    termination_reason = "horizon" if reached_horizon else ("extinction" if extinct else ("safety_stop" if safety_stop else "terminated"))
    meta = {
        "program_version": PROGRAM_VERSION, "run_id": task.run_id, "seed": int(task.seed),
        "condition": cfg.mechanism_condition, "functional_interdependence_enabled": int(cfg.functional_interdependence_enabled),
        "shared_ecological_construction_enabled": int(cfg.shared_ecological_construction_enabled),
        "completed": int(reached_horizon), "termination_reason": termination_reason, "extinct": int(extinct), "safety_stop": safety_stop,
        "generations_requested": cfg.generations, "generations_completed": int(last_generation_reached),
        "nmax": cfg.nmax, "n0": cfg.n0, "analysis_births": int(analysis_births), "runtime_seconds": runtime,
        "max_population": int(max_pop), "max_functions_per_genome": int(max_modules),
        "max_population_function_diversity_raw": int(max_function_diversity), "max_functional_genotype_diversity_raw": int(max_functional_genotype_diversity),
        "max_active_substrates": int(env.max_active_substrates), "max_token_id": int(pool.max_token), "genotype_pool_size": len(pool.keys),
        "structural_mutation_events": int(pool.structural_mutation_events), "reaction_identities_ever": int(len(pool.component_identities_ever)),
        "causal_probe_repeats": int(cfg.causal_probe_repeats), "causal_probe_alpha": float(cfg.causal_probe_alpha),
    }
    return {
        "meta": meta, "trajectory": trajectory, "channon_raw": activity._raw_rows,
        "channon_norm_present": activity._norm_present, "min_normalized_activity": activity.minimum_normalized_activity(),
        "modes": modes.rows, "causal_raw": causal_raw, "resource_dependency_raw": resource_raw,
        "component_snapshots": component_snapshots, "dependency_snapshots": dependency_snapshots,
        "dependency_edges": dependency_edges,
    }


def run_mechanism_tasks(tasks: Sequence[RunTask], workers: int, out_dir: Path) -> List[Dict[str, Any]]:
    d = out_dir / "runs"; d.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []; pending: List[RunTask] = []
    for t in tasks:
        p = d / f"{t.run_id}.json.gz"
        if p.exists():
            try:
                cached = read_json_gz(p)
                if str(cached.get("meta", {}).get("program_version", "")) == PROGRAM_VERSION:
                    results.append(cached); continue
            except Exception: pass
        pending.append(t)
    if workers <= 1:
        for i, t in enumerate(pending, 1):
            b = simulate_mechanism_task(t); write_json_gz(d / f"{t.run_id}.json.gz", b); results.append(b)
            print(f"[{i}/{len(pending)}] {t.run_id} {b['meta']['termination_reason']}; runtime={b['meta']['runtime_seconds']:.2f}s", flush=True)
    else:
        ctx = mp.get_context("spawn"); done = 0
        for start in range(0, len(pending), workers):
            batch = pending[start:start+workers]
            with ProcessPoolExecutor(max_workers=min(workers, len(batch)), mp_context=ctx) as ex:
                futs = {ex.submit(simulate_mechanism_task, t): t for t in batch}
                for fut in as_completed(futs):
                    t = futs[fut]; b = fut.result(); done += 1
                    write_json_gz(d / f"{t.run_id}.json.gz", b); results.append(b)
                    print(f"[{done}/{len(pending)}] {t.run_id} {b['meta']['termination_reason']}; runtime={b['meta']['runtime_seconds']:.2f}s", flush=True)
    return results


def _mechanism_run_index(run_id: str) -> int:
    m = re.search(r"run_(\d+)$", str(run_id))
    return int(m.group(1)) if m else -1


def aggregate_mechanism_validation(bundles: Sequence[Mapping[str, Any]], out: Path) -> Dict[str, Any]:
    status_rows=[]; run_rows=[]; assay_rows=[]; resource_rows=[]; dep_rows=[]; edge_rows=[]; causal_edge_rows=[]; hallmark_ts_rows=[]; traj_rows=[]
    bundle_map: Dict[Tuple[int,str], Mapping[str,Any]] = {}
    for b in bundles:
        meta=dict(b.get("meta",{})); rid=str(meta.get("run_id")); cond=str(meta.get("condition")); rep=_mechanism_run_index(rid)
        bundle_map[(rep,cond)] = b; status_rows.append(meta)
        finalized=finalize_causal_rows(b); ts=causal_hallmark_timeseries(b, finalized)
        for r in finalized:
            z=dict(r); z.update({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep}); assay_rows.append(z)
        for r in ts:
            z=dict(r); z.update({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep}); hallmark_ts_rows.append(z)
        for r in b.get("trajectory",[]):
            z=dict(r); z.update({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep}); traj_rows.append(z)
        for r in b.get("resource_dependency_raw",[]):
            z=dict(r); z.update({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep}); resource_rows.append(z)
            if int(z.get("strict_shared_dependency_candidate",0)) and float(z.get("mean_log_erasure_effect",0.0))>0:
                for p in [x for x in str(z.get("upstream_components_current","" )).split(";") if x]:
                    causal_edge_rows.append({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep,
                        "generation":z.get("generation"),"producer_component":p,"resource_token":z.get("input_token"),
                        "consumer_component":z.get("component"),"mean_log_erasure_effect":z.get("mean_log_erasure_effect"),
                        "p_erasure":z.get("p_erasure"),"rescue_matches_intact_fraction":z.get("rescue_matches_intact_fraction")})
        for r in b.get("dependency_snapshots",[]):
            z=dict(r); z.update({"run_id":rid,"seed":meta.get("seed"),"condition":cond,"replicate":rep}); dep_rows.append(z)
        for r in b.get("dependency_edges",[]):
            z=dict(r); z.update({"condition":cond,"replicate":rep}); edge_rows.append(z)
        requested=int(meta.get("generations_requested",0)); half=requested/2.0
        late=[r for r in ts if float(r.get("generation",0))>=half]; early=[r for r in ts if float(r.get("generation",0))<half]
        late_new=sum(int(r.get("new_positive_causal_innovations",0)) for r in late)
        late_births=max(0,int(late[-1]["analysis_births"])-int(late[0]["analysis_births"])) if len(late)>=2 else 0
        final_max=max((int(r.get("running_max_positive_causal_component_diversity",0)) for r in ts),default=0)
        cumulative=max((int(r.get("cumulative_positive_causal_innovations",0)) for r in ts),default=0)
        early_max=max((int(r.get("running_max_positive_causal_component_diversity",0)) for r in early),default=0)
        deps=[dict(r) for r in b.get("dependency_snapshots",[])]
        run_rows.append({
            "run_id":rid,"replicate":rep,"seed":meta.get("seed"),"condition":cond,"completed":meta.get("completed"),"extinct":meta.get("extinct"),
            "analysis_births":meta.get("analysis_births"),"structural_mutation_events":meta.get("structural_mutation_events"),"reaction_identities_ever":meta.get("reaction_identities_ever"),
            "n_causal_assays":len(finalized),"n_positive_causal_components":sum(int(r.get("positive_causal_effect",0)) for r in finalized),
            "late_positive_causal_innovations":late_new,"late_positive_causal_novelty_per_1000_births":1000.0*late_new/late_births if late_births>0 else 0.0,
            "early_running_max_positive_causal_diversity":early_max,"final_running_max_positive_causal_diversity":final_max,
            "cumulative_positive_causal_innovations":cumulative,
            "max_dependency_edges":max((int(r.get("n_dependency_edges",0)) for r in deps),default=0),
            "max_cross_genotype_dependency_edges":max((int(r.get("n_cross_genotype_edges",0)) for r in deps),default=0),
            "max_historically_ordered_dependency_edges":max((int(r.get("n_historically_ordered_edges",0)) for r in deps),default=0),
            "max_dependency_depth":max((int(r.get("dependency_depth",0)) for r in deps),default=0),
            "n_resource_dependency_assays":sum(1 for r in resource_rows if r.get("run_id")==rid),
            "n_strict_resource_dependency_candidates":sum(int(r.get("strict_shared_dependency_candidate",0)) for r in resource_rows if r.get("run_id")==rid),
            "n_positive_strict_resource_dependencies":sum(int(r.get("strict_shared_dependency_candidate",0)) and float(r.get("mean_log_erasure_effect",0))>0 for r in resource_rows if r.get("run_id")==rid),
        })
    # Equal evolutionary opportunity: compare conditions at the minimum final birth count
    # within each matched replicate and use the latest snapshot not exceeding that count.
    eq_rows=[]
    reps=sorted({r[0] for r in bundle_map})
    for rep in reps:
        present=[(c,bundle_map[(rep,c)]) for c in MECHANISM_CONDITIONS if (rep,c) in bundle_map]
        if len(present)<2: continue
        bstar=min(int(b.get("meta",{}).get("analysis_births",0)) for _,b in present)
        for cond,b in present:
            f=finalize_causal_rows(b); ts=causal_hallmark_timeseries(b,f)
            cand=[r for r in ts if int(r.get("analysis_births",0))<=bstar]
            if not cand: continue
            r=max(cand,key=lambda x:int(x.get("analysis_births",0)))
            traj=[x for x in b.get("trajectory",[]) if int(x.get("analysis_births",0))<=bstar]
            tr=max(traj,key=lambda x:int(x.get("analysis_births",0))) if traj else {}
            eq_rows.append({"replicate":rep,"condition":cond,"seed":b.get("meta",{}).get("seed"),"matched_birth_target":bstar,
                "observed_births":int(r.get("analysis_births",0)),"generation":int(r.get("generation",0)),
                "positive_causal_diversity":int(r.get("running_max_positive_causal_component_diversity",0)),
                "cumulative_positive_causal_innovations":int(r.get("cumulative_positive_causal_innovations",0)),
                "component_diversity":int(tr.get("component_diversity",0) or 0),
                "structural_mutation_events_cumulative":int(tr.get("structural_mutation_events_cumulative",0) or 0),
                "reaction_identities_ever":int(tr.get("reaction_identities_ever",0) or 0)})
    write_csv(out/"01_RUN_STATUS.csv",status_rows); write_csv(out/"02_MECHANISM_RUN_SUMMARY.csv",run_rows)
    write_csv(out/"03_TRAJECTORIES.csv",traj_rows); write_csv(out/"04_RESOURCE_DEPENDENCY_ASSAYS.csv",resource_rows)
    write_csv(out/"05_DEPENDENCY_NETWORK_TIMESERIES.csv",dep_rows); write_csv(out/"06_DEPENDENCY_EDGES.csv",edge_rows)
    write_csv(out/"07_CAUSALLY_VALIDATED_DEPENDENCY_EDGES.csv",causal_edge_rows); write_csv(out/"08_EQUAL_BIRTH_OPPORTUNITY.csv",eq_rows)
    write_csv(out/"09_CAUSAL_ADAPTIVE_NOVELTY_ASSAYS.csv",assay_rows); write_csv(out/"10_BEHAVIORAL_HALLMARK_TIMESERIES.csv",hallmark_ts_rows)
    return {"n_runs":len(bundles),"n_matched_replicates":len(reps),"conditions":sorted({r["condition"] for r in run_rows}),
            "n_resource_dependency_assays":len(resource_rows),"n_causally_validated_dependency_edges":len(causal_edge_rows)}


def mechanism_self_test() -> None:
    # Preserve original V12 structural tests under the FULL mechanism first.
    self_test()
    base = ModelConfig(n0=32,nmax=32,generations=10,snapshot_every=5,validation_every=5,modes_filter_generations=4,causal_probe_repeats=2)
    pool=GenotypePool(base.basal_functions); mask=(1<<base.basal_functions)-1; gid=pool.intern((mask,((0,1),(1,2))))
    state=PopulationState(np.full(32,gid,np.int32),np.full((32,base.basal_functions),base.initial_x,np.float32),np.zeros(32,np.float32),np.zeros(32,np.float32),np.full(32,-1,np.int32),np.arange(1,33,dtype=np.int64),np.zeros((32,base.modes_filter_generations),np.int64))
    full_env=ReactionEnvironment(replace(base,mechanism_condition="FULL",shared_ecological_construction_enabled=True)); eco_env=ReactionEnvironment(replace(base,mechanism_condition="ECO_MINUS",shared_ecological_construction_enabled=False))
    full_env.step(state,pool,replace(base,mechanism_condition="FULL",shared_ecological_construction_enabled=True)); eco_env.step(state,pool,replace(base,mechanism_condition="ECO_MINUS",shared_ecological_construction_enabled=False))
    assert float(np.sum(full_env.mass[1:])) > float(np.sum(eco_env.mass[1:])) + 1e-12
    print("PASS ECO_MINUS blocks shared ecological construction while preserving reaction execution",flush=True)
    print("ALL 14 MECHANISM SELF-TESTS PASSED",flush=True)

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("smoke","full"), default="full")
    p.add_argument("--workers", default="auto")
    p.add_argument("--runs", type=int, default=16, help="matched replicate identities per condition")
    p.add_argument("--conditions", default="FULL,INT_MINUS,ECO_MINUS,BOTH_MINUS")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--root-seed", type=int, default=20260823)
    p.add_argument("--out", default=None)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args=parse_args(argv)
    if args.self_test:
        mechanism_self_test(); return 0
    out=Path(args.out).expanduser().resolve() if args.out else Path.home()/"Desktop"/"REPRODUCTIVE_INTERDEPENDENCE_CAUSAL_MECHANISM_V14"
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"00_CONFIG.json",{
        "program_version":PROGRAM_VERSION,"args":vars(args),
        "conditions":{
            "FULL":"functional interdependence ON; shared ecological construction ON",
            "INT_MINUS":"pairing retained; partner functional rescue and regulatory sharing OFF; ecology ON",
            "ECO_MINUS":"interdependence ON; reactions/local cascades retained; reaction products excluded from shared environment",
            "BOTH_MINUS":"functional interdependence OFF; shared ecological construction OFF",
        },
        "primary_mechanistic_prediction":"FULL exceeds additive single-mechanism effects for adaptive novelty/diversity, and later positive-effect reactions can depend causally on historically constructed shared resources",
    })
    tasks=build_mechanism_tasks(args); w=auto_workers(args.workers,len(tasks))
    print(f"Mechanism campaign: {len(tasks)} runs ({len(set(t.cfg.mechanism_condition for t in tasks))} conditions), workers={w}",flush=True)
    bundles=run_mechanism_tasks(tasks,w,out)
    summary=aggregate_mechanism_validation(bundles,out)
    report=["# Causal mechanism validation", "", f"Program version: {PROGRAM_VERSION}", "",
        "The campaign uses matched replicate seeds across a 2 x 2 ablation of functional interdependence and shared ecological construction.",
        "INT_MINUS retains pair formation but removes partner functional rescue. ECO_MINUS retains hereditary reactions and within-genotype local routing but prevents reaction products from entering the shared environment.",
        "Historical resource-dependency assays erase the constructed input resource of a later positive-effect reaction and restore that resource in a paired rescue control.",
        "Dependency-network summaries use realized reaction components connected through currently available constructed shared resources; SCCs are condensed before dependency depth is calculated.",
        "Equal-birth summaries compare conditions at the minimum cumulative birth count within each matched replicate.", "", "```json", json.dumps(json_safe(summary),indent=2), "```", ""]
    atomic_text(out/"11_REPORT.md","\n".join(report))
    print(json.dumps(json_safe({"summary":summary,"output":str(out)}),indent=2),flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
    except Exception:
        traceback.print_exc()
        raise
