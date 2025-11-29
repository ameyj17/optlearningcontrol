#!/usr/bin/env python3
"""
Trajectory optimization for the planar quadrotor via SQP.

Implements the nine-step recipe from `project1.ipynb` (cells 12–21) and Lecture 5:
  1. Outer SQP loop with Newton iterations on the KKT conditions.
  2. Quadratic running/terminal cost and analytic gradients.
  3. Block-diagonal Hessian (Gauss–Newton approximation).
  4. Linearized dynamics constraints from forward Euler transcription.
  5. Riccati-friendly QP subproblem solved with OSQP.
  6. Filter line search balancing cost vs constraint violation.
  7. Diagnostics and plots for Part 1 deliverables.
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


# --------------------------------------------------------------------------- #
# Problem data container                                                       #
# --------------------------------------------------------------------------- #

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
    altitude_margin: float = 0.05
    slack_weight: float = 1e5  # Strong penalty while keeping QP conditioning reasonable
    qp_iters_per_step: int = 1  # Real-time SQP iterations per MPC step

DIM_X = quadrotor.DIM_STATE
DIM_U = quadrotor.DIM_CONTROL

# --------------------------------------------------------------------------- #
# Step 1: Forward Euler discrete dynamics                                     #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Steps 2–3: Cost, gradients, Hessians                                        #
# --------------------------------------------------------------------------- #

def reference_state(opts: SQPOptions, k: int) -> np.ndarray:
    return opts.state_ref(k)


def reference_control(opts: SQPOptions, k: int) -> np.ndarray:
    return opts.control_ref(k)


# def cost_grad_hess(z: np.ndarray, opts: SQPOptions) -> Tuple[float, np.ndarray, np.ndarray]:
#     N = opts.horizon
#     x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
#     u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

#     cost = 0.0
#     grad = np.zeros_like(z)
#     H_diag = np.zeros_like(z)

#     for k in range(N):
#         x_err = x_blocks[k] - reference_state(opts, k)
#         u_err = u_blocks[k] - reference_control(opts, k)

#         cost += x_err @ opts.Q @ x_err + u_err @ opts.R @ u_err
#         grad[k * DIM_X:(k + 1) * DIM_X] += 2.0 * opts.Q @ x_err
#         ctrl_slice = (N + 1) * DIM_X + k * DIM_U
#         grad[ctrl_slice:ctrl_slice + DIM_U] += 2.0 * opts.R @ u_err

#         H_diag[k * DIM_X:(k + 1) * DIM_X] += np.diag(2.0 * opts.Q)
#         H_diag[ctrl_slice:ctrl_slice + DIM_U] += np.diag(2.0 * opts.R)

#     xN_err = x_blocks[-1] - reference_state(opts, N)
#     xN_err[4] = wrap_angle(xN_err[4])  # enforce angular wrap
#     cost += xN_err @ opts.Qf @ xN_err
#     grad[-DIM_X:] += 2.0 * opts.Qf @ xN_err
#     H_diag[-DIM_X:] += np.diag(2.0 * opts.Qf)

#     # Add explicit penalty for altitude violations (negative altitude)
#     altitude_penalty_weight = 1e5
#     for k in range(N + 1):
#         altitude_violation = max(0.0, -x_blocks[k, 2])  # Only penalize negative altitude
#         cost += altitude_penalty_weight * altitude_violation ** 2
#         if altitude_violation > 0:
#             grad[k * DIM_X + 2] -= 2.0 * altitude_penalty_weight * altitude_violation
#             H_diag[k * DIM_X + 2] += 2.0 * altitude_penalty_weight

#     return cost, grad, H_diag

def cost_grad_hess(z: np.ndarray, opts: SQPOptions) -> Tuple[float, np.ndarray, np.ndarray]:
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

    cost = 0.0
    grad = np.zeros_like(z)
    H_diag = np.zeros_like(z)

    for k in range(N):
        x_err = x_blocks[k] - reference_state(opts, k)
        x_err[4] = wrap_angle(x_err[4])  # ← ADD THIS: wrap theta at every stage
        u_err = u_blocks[k] - reference_control(opts, k)

        cost += x_err @ opts.Q @ x_err + u_err @ opts.R @ u_err
        grad[k * DIM_X:(k + 1) * DIM_X] += 2.0 * opts.Q @ x_err
        ctrl_slice = (N + 1) * DIM_X + k * DIM_U
        grad[ctrl_slice:ctrl_slice + DIM_U] += 2.0 * opts.R @ u_err

        H_diag[k * DIM_X:(k + 1) * DIM_X] += np.diag(2.0 * opts.Q)
        H_diag[ctrl_slice:ctrl_slice + DIM_U] += np.diag(2.0 * opts.R)

    xN_err = x_blocks[-1] - reference_state(opts, N)
    xN_err[4] = wrap_angle(xN_err[4])
    cost += xN_err @ opts.Qf @ xN_err
    grad[-DIM_X:] += 2.0 * opts.Qf @ xN_err
    H_diag[-DIM_X:] += np.diag(2.0 * opts.Qf)

    # Remove or greatly reduce the explicit altitude penalty (already in constraint)
    # The slack penalty handles this better
    
    return cost, grad, H_diag


def build_default_options(horizon: int) -> SQPOptions:
    Q = np.diag([40.0, 4.0, 400.0, 8.0, 30.0, 6.0])  # ↑p_y: 200→400, ↑v_y: 4→8, ↓θ: 60→30
    R = np.diag([2.0, 2.0])  # ↑ control cost to avoid thrust spikes
    Qf = np.diag([320.0, 40.0, 3200.0, 80.0, 240.0, 60.0])  # Scale terminal weights similarly
    
    return SQPOptions(
        horizon=horizon,
        Q=Q,
        R=R,
        Qf=Qf,
        state_ref=default_reference_factory(horizon),
        control_ref=lambda k: default_control_reference(k),
    )


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Steps 5–6: QP subproblem and solver                                         #
# --------------------------------------------------------------------------- #

def solve_qp_subproblem(
    H_diag: np.ndarray,
    grad: np.ndarray,
    G: sp.csc_matrix,
    constr: np.ndarray,
    z: np.ndarray,
    opts: SQPOptions,
) -> np.ndarray:
    """Solve min ½ΔzᵀHΔz + gradᵀΔz s.t. GΔz = -constr, thrust bounds, and altitude bounds (with slack)."""
    N = opts.horizon
    n = H_diag.size
    num_slack = N + 1

    # Extend Hessian/gradient for slack variables
    H_extended = sp.diags(np.concatenate([np.maximum(H_diag, 1e-9), np.full(num_slack, opts.slack_weight)]))
    q_extended = np.concatenate([grad.copy(), np.zeros(num_slack)])

    # Equality: G Δz = -constr; fix Δx0 = 0 (slack doesn't affect equalities)
    A_eq_dynamics = G
    A_eq_init = sp.coo_matrix(
        (np.ones(DIM_X), (np.arange(DIM_X), np.arange(DIM_X))),
        shape=(DIM_X, n),
    ).tocsc()
    A_eq_slack = sp.coo_matrix((DIM_X + N * DIM_X, num_slack))
    A_eq = sp.hstack([sp.vstack([A_eq_dynamics, A_eq_init]), A_eq_slack]).tocsc()
    l_eq = np.concatenate([-constr, np.zeros(DIM_X)])
    u_eq = l_eq.copy()

    # Inequality: thrust bounds → linear on Δu (no slack)
    ctrl_offset = (N + 1) * DIM_X
    num_ctrl = N * DIM_U
    idx = np.arange(num_ctrl)
    A_ctrl = sp.coo_matrix(
        (np.ones(num_ctrl), (idx, ctrl_offset + idx)),
        shape=(num_ctrl, n),
    ).tocsc()
    A_ctrl_slack = sp.coo_matrix((num_ctrl, num_slack))
    A_ctrl_full = sp.hstack([A_ctrl, A_ctrl_slack]).tocsc()
    u_curr = z[ctrl_offset:]
    l_ctrl = np.full(num_ctrl, opts.thrust_min) - u_curr
    u_ctrl = np.full(num_ctrl, opts.thrust_max) - u_curr

    # Altitude constraint with slack: p_y + s >= -margin
    x_curr = z[: (N + 1) * DIM_X]
    alt_row_idx = np.arange(N + 1)
    alt_col_idx = (np.arange(N + 1) * DIM_X) + 2
    alt_data = np.ones(N + 1)
    A_alt_x = sp.coo_matrix(
        (alt_data, (alt_row_idx, alt_col_idx)),
        shape=(N + 1, n),
    ).tocsc()
    A_alt_s = sp.coo_matrix(
        (np.ones(N + 1), (alt_row_idx, alt_row_idx)),
        shape=(N + 1, num_slack),
    ).tocsc()
    A_alt_full = sp.hstack([A_alt_x, A_alt_s]).tocsc()
    l_alt = -x_curr[alt_col_idx] - opts.altitude_margin
    u_alt = np.full(N + 1, np.inf)

    # Slack non-negativity: s >= 0
    A_slack = sp.hstack([
        sp.coo_matrix((num_slack, n)),
        sp.eye(num_slack)
    ]).tocsc()
    l_slack = np.zeros(num_slack)
    u_slack = np.full(num_slack, np.inf)

    # Stack all constraints
    A = sp.vstack([A_eq, A_ctrl_full, A_alt_full, A_slack]).tocsc()
    l = np.concatenate([l_eq, l_ctrl, l_alt, l_slack])
    u = np.concatenate([u_eq, u_ctrl, u_alt, u_slack])

    solver = osqp.OSQP()
    # Increased max_iter and added regularization for better convergence
    solver.setup(
        P=H_extended, 
        q=q_extended, 
        A=A, 
        l=l, 
        u=u, 
        eps_abs=1e-5,  # Slightly relaxed for faster convergence
        eps_rel=1e-5,
        max_iter=20000,  # Increased from default ~4000
        rho=0.1,  # ADMM step size (lower = more stable but slower)
        adaptive_rho=True,  # Let OSQP adapt the step size
        polish=True,  # Polish solution for better accuracy
        verbose=False
    )
    result = solver.solve()
    
    # Accept solutions that converged or reached max_iter with reasonable progress
    if result.info.status_val not in (1, 2):
        # Check if we got a reasonable solution despite max_iter
        if result.info.status_val == -3:  # maximum iterations reached
            # If solution exists and constraint violations are reasonable, use it
            if result.x is not None and len(result.x) > 0:
                # Check if solution satisfies constraints approximately
                x_sol = result.x[:n]
                # Compute constraint violations
                eq_viol = np.linalg.norm(A_eq @ result.x - l_eq[:A_eq.shape[0]], ord=np.inf)
                if eq_viol < 1e-2:  # Accept if equality violations are small
                    print(f"[QP DEBUG] Using solution despite max_iter (eq_viol={eq_viol:.2e})")
                    return x_sol
        
        print("[QP DEBUG] status:", result.info.status)
        print("[QP DEBUG] control bounds rhs range:",
              float(l_ctrl.min()), float(l_ctrl.max()),
              float(u_ctrl.min()), float(u_ctrl.max()))
        print("[QP DEBUG] altitude rhs min/max:",
              float(l_alt.min()), float(l_alt.max()))
        print("[QP DEBUG] min altitude in guess:",
              float(x_curr[alt_col_idx].min()))
        print("[QP DEBUG] dynamics residual inf-norm:",
              float(np.linalg.norm(constr, ord=np.inf)))
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    
    # Return only the decision variable increments (drop slack)
    return result.x[:n]


# --------------------------------------------------------------------------- #
# Steps 7–8: Constraint violation & filter line search                        #
# --------------------------------------------------------------------------- #

def constraint_violation(z: np.ndarray, opts: SQPOptions) -> float:
    N = opts.horizon
    x_blocks = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_blocks = z[(N + 1) * DIM_X:].reshape(N, DIM_U)

    dyn_violation = dynamics_residual(z, opts)
    accum = np.sum(np.abs(dyn_violation))

    for k in range(N):
        accum += np.sum(np.maximum(0.0, opts.thrust_min - u_blocks[k]))
        accum += np.sum(np.maximum(0.0, u_blocks[k] - opts.thrust_max))

    # Add altitude violations (negative altitude)
    for k in range(N + 1):
        accum += max(0.0, -x_blocks[k, 2])

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
        
        # Simpler acceptance: cost OR violation improvement (like Part 1)
        if cost_trial < f_best or viol_trial < c_best:
            return z_trial, alpha, min(f_best, cost_trial), min(c_best, viol_trial), cost_trial, viol_trial
        
        alpha *= opts.line_search_rho
    return z, 0.0, f_best, c_best, cost_curr, viol_curr

# --------------------------------------------------------------------------- #
# Step 9: Outer SQP routine                                                   #
# --------------------------------------------------------------------------- #

def solve_sqp(x_init: np.ndarray, opts: SQPOptions, z0: np.ndarray | None = None):
    N = opts.horizon
    total_states = (N + 1) * DIM_X
    total_controls = N * DIM_U

    if z0 is not None:
        z = z0.copy()
        z[:DIM_X] = x_init
    else:
        x_blocks = np.zeros((N + 1, DIM_X))
        for k in range(N + 1):
            x_blocks[k] = reference_state(opts, k)
        x_blocks[0] = x_init

        u_blocks = np.zeros((N, DIM_U))
        for k in range(N):
            x_curr = x_blocks[k]
            ref_next = reference_state(opts, k + 1)

            theta = x_curr[4]
            sin_th = np.sin(theta)
            cos_th = np.cos(theta)

            A_coeffs = []
            b_vals = []

            coeff_vx = -opts.dt * sin_th / opts.mass
            A_coeffs.append(coeff_vx)
            b_vals.append(ref_next[1] - x_curr[1])

            coeff_vy = opts.dt * cos_th / opts.mass
            if abs(cos_th) > 1e-6:
                A_coeffs.append(coeff_vy)
                b_vals.append(ref_next[3] - x_curr[3] + opts.dt * opts.gravity)

            A = np.array(A_coeffs, dtype=float).reshape(-1, 1)
            b = np.array(b_vals, dtype=float)
            if A.size == 0:
                u_sum = 2.0 * opts.mass * opts.gravity
            else:
                u_sum = float(np.linalg.lstsq(A, b, rcond=None)[0])

            u_diff = (opts.inertia / (opts.arm * opts.dt)) * (ref_next[5] - x_curr[5])
            u = np.array([(u_sum + u_diff) * 0.5, (u_sum - u_diff) * 0.5])
            u = np.clip(u, opts.thrust_min + 1e-6, opts.thrust_max - 1e-6)
            u_blocks[k] = u

            x_prop = euler_step(x_curr, u, opts)
            x_blocks[k + 1] = 0.5 * x_prop + 0.5 * ref_next

        z = np.concatenate([x_blocks.reshape(-1), u_blocks.reshape(-1)])

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


# --------------------------------------------------------------------------- #
# Utility to extract trajectories and run demo                                #
# --------------------------------------------------------------------------- #

def unpack_solution(z: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    x_traj = z[: (N + 1) * DIM_X].reshape(N + 1, DIM_X)
    u_traj = z[(N + 1) * DIM_X:].reshape(N, DIM_U)
    return x_traj, u_traj


# def default_reference_factory(N: int) -> Callable[[int], np.ndarray]:
#     t_grid = np.linspace(0.0, 1.0, N + 1)

#     def ref(k: int) -> np.ndarray:
#         t = t_grid[k]
#         theta = 2.0 * np.pi * t
#         px = 1.0 * np.sin(2.0 * np.pi * t)              # ±1 m excursions

#         # Altitude profile: ramp up, hold, ramp down (prevents reference from diving)
#         if t < 0.25:
#             # Smooth ascent using cosine blend
#             py = 1.0 + 0.8 * (1.0 - np.cos(2.0 * np.pi * t / 0.25)) * 0.5
#         elif t < 0.75:
#             py = 1.8  # hold altitude during flip
#         else:
#             # Smooth descent back to 1.0 m
#             tau = (t - 0.75) / 0.25
#             py = 1.8 - 0.8 * (1.0 - np.cos(2.0 * np.pi * tau)) * 0.5

#         return np.array([px, 0.0, py, 0.0, theta, 0.0])

#     return ref
def default_reference_factory(N: int) -> Callable[[int], np.ndarray]:
    t_grid = np.linspace(0.0, 1.0, N + 1)
    
    def ref(k: int) -> np.ndarray:
        t = t_grid[k]
        theta = 2.0 * np.pi * t
        px = 1.0 * np.sin(2.0 * np.pi * t)
        py = 1.2 + 0.4 * (1.0 - np.cos(2.0 * np.pi * t))  # Simple, stays in [0.8, 1.6]
        return np.array([px, 0.0, py, 0.0, theta, 0.0])
    
    return ref

def default_control_reference(_: int) -> np.ndarray:
    thrust_hover = quadrotor.MASS * quadrotor.GRAVITY_CONSTANT / 2.0
    return np.array([thrust_hover, thrust_hover])


# def build_default_options(horizon: int) -> SQPOptions:
#     Q = np.diag([40.0, 4.0, 200.0, 4.0, 60.0, 6.0])  # Increased Q[2,2] from 40 to 200 for stronger altitude penalty
#     R = np.diag([1.2, 1.2])
#     Qf = np.diag([320.0, 40.0, 2000.0, 40.0, 500.0, 60.0])  # Increased Qf[2,2] from 320 to 2000 for terminal altitude

#     return SQPOptions(
#         horizon=horizon,
#         Q=Q,
#         R=R,
#         Qf=Qf,
#         state_ref=default_reference_factory(horizon),
#         control_ref=lambda k: default_control_reference(k),
#     )

def build_default_options(horizon: int) -> SQPOptions:
    Q = np.diag([400.0, 40.0, 400.0, 80.0, 60.0, 6.0])  
    R = np.diag([20.0, 20.0])  # ↑ control cost to avoid thrust spikes
    Qf = np.diag([3200.0, 400.0, 3200.0, 800.0, 240.0, 600.0])  # Scale terminal weights similarly
    
    return SQPOptions(
        horizon=horizon,
        Q=Q,
        R=R,
        Qf=Qf,
        state_ref=default_reference_factory(horizon),
        control_ref=lambda k: default_control_reference(k),
    )


def forward_simulate_trajectory(x0: np.ndarray, u_traj: np.ndarray, opts: SQPOptions) -> np.ndarray:
    """Forward simulate to get dynamically consistent state trajectory."""
    N = opts.horizon
    x_traj = np.zeros((N + 1, DIM_X))
    x_traj[0] = x0
    for k in range(N):
        x_traj[k + 1] = euler_step(x_traj[k], u_traj[k], opts)
    return x_traj


@dataclasses.dataclass
class SQPMPC:
    """Real-time iteration SQP MPC using one Newton step per control cycle."""
    opts: SQPOptions
    debug: bool = False
    warm_start: np.ndarray | None = None
    info_history: list[dict] = dataclasses.field(default_factory=list)

    def reset(self, x0: np.ndarray) -> None:
        """Compute a full SQP solution for the current state and use it as warm start."""
        if self.debug:
            print("[MPC] Warm-starting with full SQP solve …")
        z_guess = None
        history_accum: list[dict] = []
        for margin in (2.0, 0.5, 0.1, self.opts.altitude_margin):
            temp_opts = dataclasses.replace(self.opts, altitude_margin=margin)
            try:
                z_guess, history = solve_sqp(x0, temp_opts, z0=z_guess)
            except RuntimeError as exc:
                if self.debug:
                    print(f"[MPC] Warm-start stage with margin {margin} failed: {exc}")
                break
            x_proj = z_guess[: (self.opts.horizon + 1) * DIM_X].reshape(
                self.opts.horizon + 1, DIM_X
            )
            x_proj[:, 2] = np.maximum(x_proj[:, 2], 0.0)
            z_guess[: (self.opts.horizon + 1) * DIM_X] = x_proj.reshape(-1)
            history_accum.extend(history)

        self.warm_start = z_guess
        self.info_history = history_accum

    def _build_shifted_guess(self, x_meas: np.ndarray) -> np.ndarray:
        N = self.opts.horizon
        total_states = (N + 1) * DIM_X
        total_controls = N * DIM_U

        if self.warm_start is None:
            self.reset(x_meas)

        prev = self.warm_start
        x_prev = prev[:total_states].reshape(N + 1, DIM_X)
        u_prev = prev[total_states:].reshape(N, DIM_U)

        # Shift controls
        u_guess = np.zeros_like(u_prev)
        u_guess[:-1] = u_prev[1:]
        u_guess[-1] = self.opts.control_ref(N - 1)
        
        # Forward simulate (dynamically consistent)
        x_guess = forward_simulate_trajectory(x_meas, u_guess, self.opts)
        
        # Project altitude only if negative
        x_guess[:, 2] = np.maximum(x_guess[:, 2], 0.05)
        
        # DON'T blend with reference during MPC - trust the simulation
        # (Only blend during initial warm-start in reset())
        
        return np.concatenate([x_guess.reshape(-1), u_guess.reshape(-1)])

    def control(self, x_meas: np.ndarray, t_idx: int) -> np.ndarray:
        """Perform one SQP iteration around the warm start and return the first control."""
        z_work = self._build_shifted_guess(x_meas)

        def metric(z_trial: np.ndarray) -> tuple[float, float]:
            trial_cost, _, _ = cost_grad_hess(z_trial, self.opts)
            trial_violation = constraint_violation(z_trial, self.opts)
            return trial_cost, trial_violation

        cost_work, grad_work, H_diag_work = cost_grad_hess(z_work, self.opts)
        constr_work = dynamics_residual(z_work, self.opts)
        violation_work = constraint_violation(z_work, self.opts)

        if self.debug:
            print(
                f"[MPC] t={t_idx:03d}, iter 0: cost={cost_work:.3f}, violation={violation_work:.2e}"
            )

        alpha = 0.0
        for iter_idx in range(self.opts.qp_iters_per_step):
            G_work = dynamics_jacobian(z_work, self.opts).tocsc()
            step = None
            try:
                step = solve_qp_subproblem(
                    H_diag_work.copy(),
                    grad_work.copy(),
                    G_work,
                    constr_work.copy(),
                    z_work,
                    self.opts,
                )
            except RuntimeError:
                if self.debug:
                    print(
                        f"[MPC] t={t_idx:03d}, iter {iter_idx}: QP failed, trying relaxed margin fallback"
                    )
                try:
                    relaxed_opts = dataclasses.replace(self.opts, altitude_margin=0.5)
                    step = solve_qp_subproblem(
                        H_diag_work.copy(),
                        grad_work.copy(),
                        G_work,
                        constr_work.copy(),
                        z_work,
                        relaxed_opts,
                    )
                except RuntimeError:
                    if self.debug:
                        print(
                            f"[MPC] t={t_idx:03d}, iter {iter_idx}: Relaxed margin failed, using zero step"
                        )
                    step = np.zeros_like(z_work)

            z_candidate, alpha, _, _, cost_new, violation_new = filter_line_search(
                z_work,
                step,
                cost_work,
                violation_work,
                metric,
                self.opts,
                f_best=np.inf,
                c_best=np.inf,
            )

            if self.debug:
                print(
                    f"[MPC] t={t_idx:03d}, iter {iter_idx}: alpha={alpha:.2f}, cost={cost_new:.3f}, "
                    f"violation={violation_new:.2e}"
                )

            if alpha == 0.0:
                break

            z_work = z_candidate
            cost_work, grad_work, H_diag_work = cost_grad_hess(z_work, self.opts)
            constr_work = dynamics_residual(z_work, self.opts)
            violation_work = constraint_violation(z_work, self.opts)

        N = self.opts.horizon
        total_states = (N + 1) * DIM_X
        x_blocks = z_work[:total_states].reshape(N + 1, DIM_X)
        u_blocks = z_work[total_states:].reshape(N, DIM_U)

        min_alt = float(x_blocks[:, 2].min())
        u_min = float(u_blocks.min())
        u_max = float(u_blocks.max())

        if self.debug:
            print(
                f"[MPC] t={t_idx:03d}: final cost={cost_work:.3f}, violation={violation_work:.2e}, "
                f"min(p_y)={min_alt:.3f}, u∈[{u_min:.2f}, {u_max:.2f}]"
            )

        self.warm_start = z_work
        self.info_history.append(
            {
                "time": t_idx,
                "cost": cost_work,
                "violation": violation_work,
                "alpha": alpha,
                "min_altitude": min_alt,
            }
        )

        return u_blocks[0].copy()

# if __name__ == "__main__":
#     horizon = 80
#     opts = build_default_options(horizon)
#     x0 = np.zeros(DIM_X)

#     solution, log = solve_sqp(x0, opts)
#     x_traj, u_traj = unpack_solution(solution, horizon)

#     print(f"Final theta (rad): {x_traj[-1, 4]:.3f}")
#     print(f"Line search alphas: {[step['alpha'] for step in log if 'alpha' in step]}")
#     print(f"Constraint violation history: {[step['violation'] for step in log]}")

#     quadrotor.animate_robot(x_traj, u_traj)