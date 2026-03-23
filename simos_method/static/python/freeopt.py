"""
Lightweight compatibility shim for a subset of the gurobipy API.

This project uses only linear and mixed-integer linear features, so we map the
required interfaces to PuLP with a free solver backend.
Default priority is HiGHS (fast) with CBC fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Optional

import pulp


@dataclass(frozen=True)
class _GRBConstants:
    BINARY: str = "BINARY"
    INTEGER: str = "INTEGER"
    CONTINUOUS: str = "CONTINUOUS"

    MINIMIZE: int = 1
    MAXIMIZE: int = -1

    OPTIMAL: int = 2
    INFEASIBLE: int = 3
    INF_OR_UNBD: int = 4
    UNBOUNDED: int = 5
    OTHER: int = 6


GRB = _GRBConstants()
_SOLVER_BACKEND_CACHE: dict[tuple[str, Optional[int], Optional[int]], str] = {}


def _value_or_zero(expr) -> float:
    raw_value = pulp.value(expr)
    if raw_value is None:
        return 0.0
    return float(raw_value)


if not hasattr(pulp.LpAffineExpression, "getValue"):
    pulp.LpAffineExpression.getValue = _value_or_zero  # type: ignore[attr-defined]


def _lp_var_get_x(var: pulp.LpVariable) -> float:
    return _value_or_zero(var)


def _lp_var_get_obj(var: pulp.LpVariable) -> float:
    return float(getattr(var, "_obj_coeff", 0.0))


def _lp_var_set_obj(var: pulp.LpVariable, value) -> None:
    try:
        coeff = float(value)
    except (TypeError, ValueError):
        coeff = 0.0
    setattr(var, "_obj_coeff", coeff)
    model = getattr(var, "_free_model", None)
    if model is not None:
        model._use_var_obj_coeffs = True


if not isinstance(getattr(pulp.LpVariable, "X", None), property):
    pulp.LpVariable.X = property(_lp_var_get_x)  # type: ignore[attr-defined]


if not isinstance(getattr(pulp.LpVariable, "Obj", None), property):
    pulp.LpVariable.Obj = property(_lp_var_get_obj, _lp_var_set_obj)  # type: ignore[attr-defined]


def quicksum(terms: Iterable):
    return pulp.lpSum(list(terms))


def _to_optional_int(value):
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _solver_available(solver) -> bool:
    available_fn = getattr(solver, "available", None)
    if callable(available_fn):
        try:
            probe = available_fn()
        except Exception:
            return False
        if isinstance(probe, bool):
            return probe
        if probe is None:
            return True
        return bool(probe)
    return True


def _build_solver(msg_flag: bool):
    """
    Chooses a free solver backend.
    FREEOPT_SOLVER values:
      - auto (default): HiGHS then CBC
      - highs
      - cbc
    """
    requested = str(os.getenv("FREEOPT_SOLVER", "auto")).strip().lower()
    if requested not in {"auto", "highs", "cbc"}:
        requested = "auto"
    preferred_order = ["highs", "cbc"] if requested == "auto" else [requested]

    threads = _to_optional_int(os.getenv("FREEOPT_THREADS"))
    time_limit = _to_optional_int(os.getenv("FREEOPT_TIME_LIMIT_SEC"))
    backend_key = (requested, threads, time_limit)
    cached_backend = _SOLVER_BACKEND_CACHE.get(backend_key)
    if cached_backend is not None:
        if cached_backend == "highs":
            return pulp.HiGHS(msg=msg_flag, threads=threads, timeLimit=time_limit)
        if cached_backend == "cbc":
            return pulp.PULP_CBC_CMD(msg=msg_flag, threads=threads, timeLimit=time_limit)

    for solver_name in preferred_order:
        try:
            if solver_name == "highs":
                candidate = pulp.HiGHS(msg=msg_flag, threads=threads, timeLimit=time_limit)
            elif solver_name == "cbc":
                candidate = pulp.PULP_CBC_CMD(msg=msg_flag, threads=threads, timeLimit=time_limit)
            else:
                continue
            if _solver_available(candidate):
                _SOLVER_BACKEND_CACHE[backend_key] = solver_name
                return candidate
        except Exception:
            continue

    raise RuntimeError(
        "No available free solver backend found. Install `highspy` (recommended) "
        "or ensure CBC is available via PuLP."
    )


class Model:
    def __init__(self, name: str = "Model"):
        self._problem = pulp.LpProblem(name, pulp.LpMinimize)
        self._vars = []
        self._params = {}
        self._explicit_objective_set = False
        self._objective_expr = 0.0
        self._objective_sense = GRB.MINIMIZE
        self._use_var_obj_coeffs = False
        self.status = GRB.OTHER

    def setParam(self, name: str, value) -> None:
        self._params[name] = value

    def addVar(self, lb: float = 0.0, ub: Optional[float] = None, obj: float = 0.0,
               vtype: Optional[str] = None, name: Optional[str] = None):
        if vtype == GRB.BINARY:
            category = pulp.LpBinary
        elif vtype == GRB.INTEGER:
            category = pulp.LpInteger
        else:
            category = pulp.LpContinuous

        variable = pulp.LpVariable(name=name, lowBound=lb, upBound=ub, cat=category)
        setattr(variable, "_obj_coeff", float(obj) if obj is not None else 0.0)
        setattr(variable, "_free_model", self)
        self._vars.append(variable)
        if abs(getattr(variable, "_obj_coeff", 0.0)) > 0.0:
            self._use_var_obj_coeffs = True
        return variable

    def addConstr(self, constraint, name: Optional[str] = None):
        if name:
            self._problem += constraint, name
        else:
            self._problem += constraint
        return constraint

    def setObjective(self, expr, sense: int = GRB.MINIMIZE) -> None:
        self._explicit_objective_set = True
        self._use_var_obj_coeffs = False
        self._objective_expr = expr
        self._objective_sense = sense
        self._problem.sense = pulp.LpMinimize if sense == GRB.MINIMIZE else pulp.LpMaximize
        self._problem.objective = expr if expr is not None else 0.0

    def optimize(self) -> None:
        if self._use_var_obj_coeffs:
            self._problem.objective = pulp.lpSum(
                float(getattr(var, "_obj_coeff", 0.0)) * var
                for var in self._vars
            )
            self._problem.sense = (
                pulp.LpMinimize if self._objective_sense == GRB.MINIMIZE else pulp.LpMaximize
            )
        elif self._explicit_objective_set:
            self._problem.objective = self._objective_expr if self._objective_expr is not None else 0.0
            self._problem.sense = (
                pulp.LpMinimize if self._objective_sense == GRB.MINIMIZE else pulp.LpMaximize
            )
        else:
            self._problem.objective = 0.0
            self._problem.sense = pulp.LpMinimize

        message_flag = bool(self._params.get("OutputFlag", 0))
        solver = _build_solver(message_flag)
        self._problem.solve(solver)

        pulp_status = self._problem.status
        if pulp_status == pulp.LpStatusOptimal:
            self.status = GRB.OPTIMAL
        elif pulp_status == pulp.LpStatusInfeasible:
            self.status = GRB.INFEASIBLE
        elif pulp_status == pulp.LpStatusUnbounded:
            self.status = GRB.UNBOUNDED
        else:
            self.status = GRB.INF_OR_UNBD


def warmup_solver_backend() -> bool:
    """
    Performs a tiny solve to preload the selected backend and reduce first-use latency.
    Returns True if warmup solved optimally, else False.
    """
    try:
        model = Model("Solver_Warmup")
        x_var = model.addVar(lb=0.0, name="warmup_x")
        model.addConstr(x_var >= 1.0, "warmup_lb")
        model.setObjective(x_var, GRB.MINIMIZE)
        model.optimize()
        return model.status == GRB.OPTIMAL
    except Exception:
        return False
