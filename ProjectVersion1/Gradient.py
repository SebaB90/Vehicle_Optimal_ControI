#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import Dynamics as dyn


#define params
ns = dyn.ns  # number of states
ni = dyn.ni  # number of inputs
dt = dyn.dt  # sample time

TT = dyn.tf  # Final time in seconds
T = dyn.TT  # Number of discrete-time samples
T_mid = dyn.TT_mid
term_cond = 1e-6        #terminal condition

# ARMIJO PARAMETERS
cc = 0.5
beta = 0.7
armijo_maxiters = 20    # number of Armijo iterations
stepsize_0 = 1          # initial stepsize


def cost(xx, uu, xx_ref, uu_ref, Q, R):

    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]

    l = 0.5*(xx - xx_ref).T@Q@(xx - xx_ref) + 0.5*(uu - uu_ref).T@R@(uu - uu_ref)

    lx = Q@(xx - xx_ref)
    lu = R@(uu - uu_ref)

    return l.squeeze(), lx, lu

def cost_f(xx, xx_ref, QT):

    xx = xx[:,None]
    xx_ref = xx_ref[:,None]

    lT = 0.5*(xx - xx_ref).T@QT@(xx - xx_ref)

    lTx = QT@(xx - xx_ref)

    return lT.squeeze(), lTx

def Gradient (xx, uu, xx_ref, uu_ref, Q, R, QT, max_iters):

    # arrays to store data
    lmbd = np.zeros((ns, T, max_iters))    # lambdas - costate seq.
    deltau = np.zeros((ni,T, max_iters))   # Du - descent direction
    dJ = np.zeros((ni,T, max_iters))       # DJ - gradient of J wrt u
    JJ = np.zeros(max_iters)                # collect cost
    descent = np.zeros(max_iters)           # collect descent direction
    descent_arm = np.zeros(max_iters)       # collect descent direction
    x0 = np.copy(xx_ref[:,0])

    kk = 0

    for kk in range(max_iters-1):

        JJ[kk] = 0

        # calculate cost
        for tt in range(T-1):
            temp_cost = cost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
            JJ[kk] += temp_cost

        temp_cost = cost_f(xx[:,-1,kk], xx_ref[:,-1], QT)[0]
        JJ[kk] += temp_cost

        # Descent direction calculation
        lmbd_temp = cost_f(xx[:,T-1,kk], xx_ref[:,T-1], QT)[1]
        lmbd[:,T-1,kk] = lmbd_temp.squeeze()

        for tt in reversed(range(T-1)):                        # integration backward in time

            at, bt = cost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[1:]
            fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]

            At = fx.T
            Bt = fu.T

            lmbd_temp = At.T@lmbd[:,tt+1,kk][:,None] + at       # costate equation
            dJ_temp = Bt.T@lmbd[:,tt+1,kk][:,None] + bt         # gradient of J wrt u
            deltau_temp = - dJ_temp

            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()
            deltau[:,tt,kk] = deltau_temp.squeeze()

            descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]
            descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk]

        # Stepsize selection - ARMIJO
        stepsizes = []  # list of stepsizes
        costs_armijo = []

        stepsize = stepsize_0

        for ii in range(armijo_maxiters):

            # temp solution update

            xx_temp = np.zeros((ns,T))
            uu_temp = np.zeros((ni,T))

            xx_temp[:,0] = x0

            for tt in range(T-1):
                uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(T-1):
                temp_cost = cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
                JJ_temp += temp_cost

            temp_cost = cost_f(xx_temp[:,-1], xx_ref[:,-1], QT)[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)                              # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100*JJ[kk]]))      # save the cost associated to the stepsize

            if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
                # update the stepsize
                stepsize = beta*stepsize

            else:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
                break

        # Armijo plot

        steps = np.linspace(0,stepsize_0,int(2e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]

            # temp solution update

            xx_temp = np.zeros((ns,T))
            uu_temp = np.zeros((ni,T))

            xx_temp[:,0] = x0

            for tt in range(T-1):
                uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(T-1):
                temp_cost = cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
                JJ_temp += temp_cost

            temp_cost = cost_f(xx_temp[:,-1], xx_ref[:,-1], QT)[0]
            JJ_temp += temp_cost

            costs[ii] = np.min([JJ_temp, 100*JJ[kk]])

        print('\nJJ(k)\n',JJ[kk],'\ndescent arm\n',descent_arm[kk],'\ndJ\n',dJ[:,:,kk],'\ndeltau\n',deltau[:,:,kk])
        plt.figure(1)
        plt.clf()
        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()
        plt.show()

        # Update the current solution

        xx_temp = np.zeros((ns,T))
        uu_temp = np.zeros((ni,T))

        xx_temp[:,0] = x0

        for tt in range(T-1):
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition

        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

        if descent[kk] <= term_cond:
            max_iters = kk
            break

    return xx, uu, descent, JJ