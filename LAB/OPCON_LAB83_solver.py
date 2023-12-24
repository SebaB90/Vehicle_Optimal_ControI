# Model Predictive Control for Linear Systems
# Solvers
# OPTCON 2022
# Lorenzo Sforni
# 20 Dec 2022
#

import numpy as np
import cvxpy as cp


def unconstrained_lqr(AA, BB, QQ, RR, QQf, xx0, T_hor = 100):
    """
        LQR - given init condition and time horizon, optimal state-input trajectory

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xx0: initial condition
          - T: time horizon
    """

    xx0 = xx0.squeeze()

    ns, ni = BB.shape

    xx_lqr = cp.Variable((ns, T_hor))
    uu_lqr = cp.Variable((ni, T_hor))

    cost = 0
    constr = []

    for tt in range(T_hor-1):
        cost += cp.quad_form(xx_lqr[:,tt], QQ) + cp.quad_form(uu_lqr[:,tt], RR)
        constr += [xx_lqr[:,tt+1] == AA@xx_lqr[:,tt] + BB@uu_lqr[:,tt]]
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_lqr[:,T_hor-1], QQf)
    constr += [xx_lqr[:,0] == xx0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! ")

    return xx_lqr.value, uu_lqr.value


def linear_mpc(AA, BB, QQ, RR, QQf, xxt, umax = 1, umin = -1, x1_max = 20, x1_min = -20, x2_max = 20, x2_min = -20,  T_pred = 5):
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon

        Returns
          - u_t: input to be applied at t
          - xx, uu predicted trajectory

    """

    xxt = xxt.squeeze()

    ns, ni = BB.shape

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []

    for tt in range(T_pred-1):
        cost += cp.quad_form(xx_mpc[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt], RR)
        constr += [xx_mpc[:,tt+1] == AA@xx_mpc[:,tt] + BB@uu_mpc[:,tt], # dynamics constraint
                uu_mpc[:,tt] <= umax, # other constraints
                uu_mpc[:,tt] >= umin,
                xx_mpc[0,tt] <= x1_max,
                xx_mpc[0,tt] >= x1_min,
                xx_mpc[1,tt] <= x2_max,
                xx_mpc[1,tt] >= x2_min]
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)
    constr += [xx_mpc[:,0] == xxt]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value