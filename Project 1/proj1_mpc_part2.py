#!/usr/bin/env python3
"""
Enhanced SQP MPC for Planar Quadrotor with Slack Variables for Robustness.

Fixes:
- Adds slack variables for altitude constraints to prevent infeasibility.
- Increases horizon and iterations for better convergence.
- Adjusts cost weights to prioritize altitude and looping.
- Improves warm-start with forward simulation and altitude projection.
- Ensures one full loop over simulation time.
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
    max_iters: int = 100  # Increased from 80
    tol: float = 1e-4
    line_search_rho: float = 0.5
    min_alpha: float = 1e-6
    altitude_margin: float = 0.0  # No margin for strict altitude
    slack_weight: float = 1e6  # Higher penalty for slack variables
    qp_iters_per_step: int = 80  # For MPC


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

    # No additional altitude penalty here, handled by slacks in QP

    return cost, grad, H_diag


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
    """Solve min ½ΔzᵀHΔz + gradᵀΔz + slack_penalty s.t. constraints with slack."""
    N = opts.horizon
    n = H_diag.size
    num_slack = N + 1

    # Extend for slack
    H_extended = sp.diags(np.concatenate([np.maximum(H_diag, 1e-9), np.full(num_slack, opts.slack_weight)]))
    q_extended = np.concatenate([grad.copy(), np.zeros(num_slack)])

    # Equality: dynamics + init
    A_eq_dynamics = G
    A_eq_init = sp.coo_matrix(
        (np.ones(DIM_X), (np.arange(DIM_X), np.arange(DIM_X))),
        shape=(DIM_X, n),
    ).tocsc()
    A_eq_slack = sp.coo_matrix((DIM_X + N * DIM_X, num_slack))
    A_eq = sp.hstack([sp.vstack([A_eq_dynamics, A_eq_init]), A_eq_slack]).tocsc()
    l_eq = np.concatenate([-constr, np.zeros(DIM_X)])
    u_eq = l_eq.copy()

    # Thrust bounds
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

    # Altitude with slack: p_y + Δp_y + s >= -margin
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

    # Slack non-negativity
    A_slack = sp.hstack([
        sp.coo_matrix((num_slack, n)),
        sp.eye(num_slack)
    ]).tocsc()
    l_slack = np.zeros(num_slack)
    u_slack = np.full(num_slack, np.inf)

    A = sp.vstack([A_eq, A_ctrl_full, A_alt_full, A_slack]).tocsc()
    l = np.concatenate([l_eq, l_ctrl, l_alt, l_slack])
    u = np.concatenate([u_eq, u_ctrl, u_alt, u_slack])

    solver = osqp.OSQP()
    solver.setup(
        P=H_extended,
        q=q_extended,
        A=A,
        l=l,
        u=u,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=20000,
        verbose=False
    )
    result = solver.solve()
    if result.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    return result.x[:n]  # Drop slack


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

    # Altitude violations
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
        accept = False
        if viol_trial < viol_curr - 1e-6:
            accept = True
        elif cost_trial < cost_curr - 1e-6 and viol_trial <= viol_curr * (1.0 + 5e-3):
            accept = True
        if accept:
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
            u_blocks[k] = reference_control(opts, k)

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
            print("Line search failed.")
            break
        z = z_new

    return z, history


# --------------------------------------------------------------------------- #
# MPC Class                                                                   #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SQPMPC:
    opts: SQPOptions
    debug: bool = False
    warm_start: np.ndarray | None = None
    info_history: list[dict] = dataclasses.field(default_factory=list)

    def reset(self, x0: np.ndarray) -> None:
        if self.debug:
            print("[MPC] Warm-starting...")
        z_guess = None
        history_accum: list[dict] = []
        for margin in (2.0, 0.5, 0.1, self.opts.altitude_margin):
            temp_opts = dataclasses.replace(self.opts, altitude_margin=margin)
            try:
                z_guess, history = solve_sqp(x0, temp_opts, z0=z_guess)
            except RuntimeError:
                break
            x_proj = z_guess[: (self.opts.horizon + 1) * DIM_X].reshape(
                self.opts.horizon + 1, DIM_X
            )
            x_proj[:, 2] = np.maximum(x_proj[:, 2], 0.1)  # Project altitude
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

        # Forward simulate
        x_guess = forward_simulate_trajectory(x_meas, u_guess, self.opts)

        # Project altitude
        x_guess[:, 2] = np.maximum(x_guess[:, 2], 0.3)

        # Blend with reference
        ref_stack = np.vstack([reference_state(self.opts, k) for k in range(N + 1)])
        x_guess = 0.5 * x_guess + 0.5 * ref_stack

        return np.concatenate([x_guess.reshape(-1), u_guess.reshape(-1)])

    def control(self, x_meas: np.ndarray, t_idx: int) -> np.ndarray:
        z_work = self._build_shifted_guess(x_meas)

        def metric(z_trial: np.ndarray) -> tuple[float, float]:
            trial_cost, _, _ = cost_grad_hess(z_trial, self.opts)
            trial_violation = constraint_violation(z_trial, self.opts)
            return trial_cost, trial_violation

        cost_work, grad_work, H_diag_work = cost_grad_hess(z_work, self.opts)
        constr_work = dynamics_residual(z_work, self.opts)
        violation_work = constraint_violation(z_work, self.opts)

        # Detailed debug prints
        if self.debug:
            print(f"[MPC DEBUG] t={t_idx:03d}: Measured state: px={x_meas[0]:.3f}, vx={x_meas[1]:.3f}, py={x_meas[2]:.3f}, vy={x_meas[3]:.3f}, theta={x_meas[4]:.3f}, omega={x_meas[5]:.3f}")
            ref_state = reference_state(self.opts, 0)
            print(f"[MPC DEBUG] t={t_idx:03d}: Reference at k=0: px={ref_state[0]:.3f}, vx={ref_state[1]:.3f}, py={ref_state[2]:.3f}, vy={ref_state[3]:.3f}, theta={ref_state[4]:.3f}, omega={ref_state[5]:.3f}")
            print(f"[MPC DEBUG] t={t_idx:03d}: Dynamics residual inf-norm: {np.linalg.norm(constr_work, ord=np.inf):.2e}")
            print(f"[MPC DEBUG] t={t_idx:03d}, iter 0: cost={cost_work:.3f}, violation={violation_work:.2e}")

        alpha = 0.0
        for iter_idx in range(self.opts.qp_iters_per_step):
            G_work = dynamics_jacobian(z_work, self.opts).tocsc()
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
                print(f"[MPC] t={t_idx:03d}, iter {iter_idx}: alpha={alpha:.2f}, cost={cost_new:.3f}, violation={violation_new:.2e}")

            if alpha == 0.0:
                if self.debug:
                    print(f"[MPC DEBUG] t={t_idx:03d}: Line search failed, stopping iterations")
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
            print(f"[MPC DEBUG] t={t_idx:03d}: Predicted trajectory - Final theta: {x_blocks[-1, 4]:.3f}, Final py: {x_blocks[-1, 2]:.3f}")
            print(f"[MPC DEBUG] t={t_idx:03d}: Control inputs range: u1={u_blocks[0, 0]:.2f}, u2={u_blocks[0, 1]:.2f}")
            print(f"[MPC] t={t_idx:03d}: final cost={cost_work:.3f}, violation={violation_work:.2e}, min(p_y)={min_alt:.3f}, u∈[{u_min:.2f}, {u_max:.2f}]")

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


def forward_simulate_trajectory(x0: np.ndarray, u_traj: np.ndarray, opts: SQPOptions) -> np.ndarray:
    N = opts.horizon
    x_traj = np.zeros((N + 1, DIM_X))
    x_traj[0] = x0
    for k in range(N):
        x_traj[k + 1] = euler_step(x_traj[k], u_traj[k], opts)
    return x_traj


# --------------------------------------------------------------------------- #
# Reference and Options                                                      #
# --------------------------------------------------------------------------- #

def default_reference_factory(N: int, total_time: float = 2.0) -> Callable[[int], np.ndarray]:
    """Periodic reference for looping over total_time seconds."""

    def ref(k: int) -> np.ndarray:
        t = (k * quadrotor.DT) % total_time  # Periodic time
        theta = 2.0 * np.pi * (t / total_time)  # Full 2π over total_time
        px = 1.0 * np.sin(2.0 * np.pi * (t / total_time))
        py = 1.2 + 0.4 * (1.0 - np.cos(2.0 * np.pi * (t / total_time)))  # Altitude profile
        return np.array([px, 0.0, py, 0.0, theta, 0.0])

    return ref


def default_control_reference(_: int) -> np.ndarray:
    thrust_hover = quadrotor.MASS * quadrotor.GRAVITY_CONSTANT / 2.0
    return np.array([thrust_hover, thrust_hover])


def build_default_options(horizon: int) -> SQPOptions:
    Q = np.diag([100.0, 4.0, 200.0, 4.0, 150.0, 6.0])  # Balanced weights
    R = np.diag([3.0, 3.0])  # Moderate control penalty
    Qf = np.diag([320.0, 40.0, 2000.0, 40.0, 500.0, 60.0])  # Higher Qf[2] for terminal altitude

    return SQPOptions(
        horizon=horizon,
        Q=Q,
        R=R,
        Qf=Qf,
        state_ref=default_reference_factory(horizon),
        control_ref=lambda k: default_control_reference(k),
    )


# Example usage in notebook
# mpc_horizon = 50  # For 2s simulation
# mpc_opts = build_default_options(mpc_horizon)
# mpc = SQPMPC(mpc_opts, debug=True)
# # Then use as in notebook
