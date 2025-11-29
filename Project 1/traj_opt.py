#!/usr/bin/env python3
"""
Trajectory optimization for the planar quadrotor via SQP.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import osqp  # fast sparse QP solver (pip install osqp)
except ImportError as exc:
    raise ImportError("Please install OSQP (pip install osqp)") from exc

import quadrotor


#   #
# Problem data container                                                       #
#   #

@dataclasses.dataclass
class SQPOptions:
    horizon: int
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    state_ref: Callable[[int], np.ndarray]
    control_ref: Callable[[int], np.ndarray]
    dt: float = quadrotor.DT
    mass: float = quadrotor.MASS
    inertia: float = quadrotor.INERTIA
    arm: float = quadrotor.LENGTH
    gravity: float = quadrotor.GRAVITY_CONSTANT
    thrust_min: float = 0.0
    thrust_max: float = 10.0
    max_iters: int = 80
    tol: float = 1e-4
    line_search_rho: float = 0.5
    min_alpha: float = 1e-6


DIM_X = quadrotor.DIM_STATE
DIM_U = quadrotor.DIM_CONTROL


#   #
# Step 1: Forward Euler discrete dynamics                                     #
#   #

def euler_step(x: np.ndarray, u: np.ndarray, opts: SQPOptions) -> np.ndarray:
    """x_{k+1} = x_k + Δt f(x_k,u_k)."""
    px, vx, py, vy, theta, omega = x
    u1, u2 = u
    u_sum = u1 + u2
    u_diff = u1 - u2

    f = np.array([
        vx,
        -(u_sum / opts.mass) * np.sin(theta),
        vy,
        (u_sum / opts.mass) * np.cos(theta) - opts.gravity,
        omega,
        (opts.arm / opts.inertia) * u_diff,
    ])
    return x + opts.dt * f


def dynamics_residual(z: np.ndarray, opts: SQPOptions) -> np.ndarray:
    """Stack c_k(x_k,u_k,x_{k+1}) = x_{k+1} - step(x_k,u_k) for k=0..N-1."""
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)
    residuals = []
    for k in range(N):
        residuals.append(euler_step(x_blocks[k], u_blocks[k], opts) - x_blocks[k + 1])
    return np.concatenate(residuals)


def dynamics_jacobian(z: np.ndarray, opts: SQPOptions) -> sp.coo_matrix:
    """Sparse Jacobian G(z) = ∂c/∂z with banded block structure."""
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

    rows, cols, data = [], [], []

    def add_block(row_base: int, col_base: int, block: np.ndarray) -> None:
        nz_rows, nz_cols = np.nonzero(block)
        rows.extend(row_base + nz_rows)
        cols.extend(col_base + nz_cols)
        data.extend(block[nz_rows, nz_cols])

    I_x = np.eye(DIM_X)
    for k in range(N):
        theta = x_blocks[k, 4]
        u1, u2 = u_blocks[k]
        u_sum = u1 + u2
        sin_th, cos_th = np.sin(theta), np.cos(theta)

        A = I_x.copy()
        A[1, 4] += -opts.dt * (u_sum / opts.mass) * cos_th
        A[3, 4] += -opts.dt * (u_sum / opts.mass) * sin_th
        A[4, 5] += opts.dt

        B = np.zeros((DIM_X, DIM_U))
        B[1, :] = -opts.dt * sin_th / opts.mass
        B[3, :] = opts.dt * cos_th / opts.mass
        B[5, 0] = opts.dt * opts.arm / opts.inertia
        B[5, 1] = -opts.dt * opts.arm / opts.inertia

        row_offset = k * DIM_X
        add_block(row_offset, k * DIM_X, A)
        add_block(row_offset, (k + 1) * DIM_X, -I_x)
        ctrl_col = (N + 1) * DIM_X + k * DIM_U
        add_block(row_offset, ctrl_col, B)

    total_cols = (N + 1) * DIM_X + N * DIM_U
    return sp.coo_matrix((data, (rows, cols)), shape=(N * DIM_X, total_cols))


#   #
# Steps 2–3: Cost, gradients, Hessians                                        #

def reference_state(opts: SQPOptions, k: int) -> np.ndarray:
    return opts.state_ref(k)


def reference_control(opts: SQPOptions, k: int) -> np.ndarray:
    return opts.control_ref(k)


def cost_grad_hess(z: np.ndarray, opts: SQPOptions) -> Tuple[float, np.ndarray, np.ndarray]:
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

    cost = 0.0
    grad = np.zeros_like(z)
    H_diag = np.zeros_like(z)

    for k in range(N):
        x_err = x_blocks[k] - reference_state(opts, k)
        u_err = u_blocks[k] - reference_control(opts, k)

        cost += x_err @ opts.Q @ x_err + u_err @ opts.R @ u_err
        grad[k * DIM_X:(k + 1) * DIM_X] += 2.0 * opts.Q @ x_err
        ctrl_slice = (N + 1) * DIM_X + k * DIM_U
        grad[ctrl_slice:ctrl_slice + DIM_U] += 2.0 * opts.R @ u_err

        H_diag[k * DIM_X:(k + 1) * DIM_X] += np.diag(2.0 * opts.Q)
        H_diag[ctrl_slice:ctrl_slice + DIM_U] += np.diag(2.0 * opts.R)

    xN_err = x_blocks[-1] - reference_state(opts, N)
    xN_err[4] = wrap_angle(xN_err[4])  # enforce angular wrap
    cost += xN_err @ opts.Qf @ xN_err
    grad[-DIM_X:] += 2.0 * opts.Qf @ xN_err
    H_diag[-DIM_X:] += np.diag(2.0 * opts.Qf)

    return cost, grad, H_diag


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


#   #
# Steps 5–6: QP subproblem and solver                                         #
#   #

def solve_qp_subproblem(
    H_diag: np.ndarray,
    grad: np.ndarray,
    G: sp.csc_matrix,
    constr: np.ndarray,
    z: np.ndarray,
    opts: SQPOptions,
) -> np.ndarray:
    """Solve min ½ΔzᵀHΔz + gradᵀΔz s.t. GΔz = -constr, thrust bounds, and altitude bounds."""
    N = opts.horizon
    n = H_diag.size

    P = sp.diags(np.maximum(H_diag, 1e-9))
    q = grad.copy()

    # Equality: G Δz = -constr; fix Δx0 = 0
    A_eq_dynamics = G
    A_eq_init = sp.coo_matrix(
        (np.ones(DIM_X), (np.arange(DIM_X), np.arange(DIM_X))),
        shape=(DIM_X, n),
    ).tocsc()
    A_eq = sp.vstack([A_eq_dynamics, A_eq_init]).tocsc()
    l_eq = np.concatenate([-constr, np.zeros(DIM_X)])
    u_eq = l_eq.copy()

    # Inequality: thrust bounds → linear on Δu
    ctrl_offset = (N + 1) * DIM_X
    num_ctrl = N * DIM_U
    idx = np.arange(num_ctrl)
    A_ctrl = sp.coo_matrix(
        (np.ones(num_ctrl), (idx, ctrl_offset + idx)),
        shape=(num_ctrl, n),
    ).tocsc()
    u_curr = z[ctrl_offset:]
    l_ctrl = np.full(num_ctrl, opts.thrust_min) - u_curr
    u_ctrl = np.full(num_ctrl, opts.thrust_max) - u_curr

    A = sp.vstack([A_eq, A_ctrl]).tocsc()
    l = np.concatenate([l_eq, l_ctrl])
    u = np.concatenate([u_eq, u_ctrl])

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
    result = solver.solve()
    if result.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    return result.x


#   #
# Steps 7–8: Constraint violation & filter line search                        #
#   #

def constraint_violation(z: np.ndarray, opts: SQPOptions) -> float:
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

    dyn_violation = dynamics_residual(z, opts)
    accum = np.sum(np.abs(dyn_violation))

    for k in range(N):
        accum += np.sum(np.maximum(0.0, opts.thrust_min - u_blocks[k]))
        accum += np.sum(np.maximum(0.0, u_blocks[k] - opts.thrust_max))

    accum += np.linalg.norm(x_blocks[0] - reference_state(opts, 0), ord=1)
    return accum


def filter_line_search(
    z: np.ndarray,
    step: np.ndarray,
    cost_curr: float,
    viol_curr: float,
    metric_fn: Callable[[np.ndarray], Tuple[float, float]],
    opts: SQPOptions,
    f_best: float,
    c_best: float,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    alpha = 1.0
    while alpha > opts.min_alpha:
        z_trial = z + alpha * step
        cost_trial, viol_trial = metric_fn(z_trial)
        if cost_trial < f_best or viol_trial < c_best:
            return z_trial, alpha, min(f_best, cost_trial), min(c_best, viol_trial), cost_trial, viol_trial
        alpha *= opts.line_search_rho
    return z, 0.0, f_best, c_best, cost_curr, viol_curr


#   #
# Step 9: Outer SQP routine                                                   #
#   #

def solve_sqp(x_init: np.ndarray, opts: SQPOptions):
    N = opts.horizon
    z = np.zeros((N + 1) * DIM_X + N * DIM_U)
    z[:DIM_X] = x_init
    for k in range(1, N + 1):
        z[k * DIM_X:(k + 1) * DIM_X] = reference_state(opts, k)
    for k in range(N):
        ctrl_slice = (N + 1) * DIM_X + k * DIM_U
        z[ctrl_slice:ctrl_slice + DIM_U] = reference_control(opts, k)

    history = []
    f_best = np.inf
    c_best = np.inf

    for it in range(opts.max_iters):
        cost, grad, H_diag = cost_grad_hess(z, opts)
        constr = dynamics_residual(z, opts)
        viol = constraint_violation(z, opts)

        history.append({"iter": it, "cost": cost, "violation": viol})

        if viol < opts.tol and np.linalg.norm(grad, ord=np.inf) < opts.tol:
            break

        G = dynamics_jacobian(z, opts).tocsc()
        step = solve_qp_subproblem(H_diag, grad, G, constr, z, opts)


        def metric(z_trial: np.ndarray) -> Tuple[float, float]:
            c_val, _, _ = cost_grad_hess(z_trial, opts)
            v_val = constraint_violation(z_trial, opts)
            return c_val, v_val

        z_new, alpha, f_best, c_best, cost_new, viol_new = filter_line_search(
            z, step, cost, viol, metric, opts, f_best, c_best
        )
        history[-1].update({"alpha": alpha, "cost_new": cost_new, "violation_new": viol_new})

        if alpha == 0.0:
            print("Line search failed to find acceptable step.")
            break
        z = z_new

    return z, history


#   #
# Utility to extract trajectories and run demo                                #
#   #

def unpack_solution(z: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    x_traj = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_traj = z[(N + 1) * DIM_X:].reshape(N, DIM_U)
    return x_traj, u_traj


def default_reference_factory(N: int) -> Callable[[int], np.ndarray]:
    t_grid = np.linspace(0.0, 1.0, N + 1)

    def ref(k: int) -> np.ndarray:
        t = t_grid[k]
        theta = 2.0 * np.pi * t
        px = 1.0 * np.sin(2.0 * np.pi * t)              # ±1 m excursions
        py = 1.2 + 0.4 * (1.0 - np.cos(2.0 * np.pi * t))  # keep y between 0.8–1.6 m
        return np.array([px, 0.0, py, 0.0, theta, 0.0])

    return ref


def default_control_reference(_: int) -> np.ndarray:
    thrust_hover = quadrotor.MASS * quadrotor.GRAVITY_CONSTANT / 2.0
    return np.array([thrust_hover, thrust_hover])


def build_default_options(horizon: int) -> SQPOptions:
    Q = np.diag([400.0, 40.0, 400.0, 80.0, 60.0, 6.0])  # ↑p_y: 200→400, ↑v_y: 4→8, ↓θ: 60→30
    R = np.diag([20.0, 20.0])  # ↑ control cost to avoid thrust spikes
    Qf = np.diag([3200.0, 400.0, 3200.0, 800.0, 240.0, 600.0]) 

    return SQPOptions(
        horizon=horizon,
        Q=Q,
        R=R,
        Qf=Qf,
        state_ref=default_reference_factory(horizon),
        control_ref=lambda k: default_control_reference(k),
    )
