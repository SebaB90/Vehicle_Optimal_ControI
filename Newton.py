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
ns = dyn.ns              #number of states
ni = dyn.ni              #number of inputs
dt = dyn.dt           #sample time

TT = dyn.TT             #discrete time samples
T = dyn.T               #time instants
T_mid = dyn.T_mid       #half time
term_cond = 1e-6        #terminal condition

# ARMIJO PARAMETERS
cc = 0.5
beta = 0.7
armijo_maxiters = 20    # number of Armijo iterations
stepsize_0 = 1          # initial stepsize


def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  if qqin is not None or rrin is not None or qqfin is not None:
    augmented = True
    print("Augmented term!")

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0
  
  PP[:,:,-1] = QQf
  pp[:,-1] = qqf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()

  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    # for other purposes we could add a regularization step here...

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()


  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
    xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return KK, sigma, PP, xxout, uuout





def cost(xx, uu, xx_ref, uu_ref, Q, R):

    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]

    l = 0.5*(xx - xx_ref).T@Q@(xx - xx_ref) + 0.5*(uu - uu_ref).T@R@(uu - uu_ref)

    lx = Q@(xx - xx_ref)
    lu = R@(uu - uu_ref)

    lxx = Q
    luu = R
    lux = np.zeros((ns, ni))
    lxu = np.zeros((ni, ns))

    gradient = np.concatenate([lx, lu], axis=0)
    hessian = np.block([[lxx, lux], [lxu, luu]])

    return l.squeeze(), gradient.squeeze(), hessian.squeeze() 

def cost_f(xx, xx_ref, QT):
    
    xx = xx[:,None]
    xx_ref = xx_ref[:,None]

    lT = 0.5*(xx - xx_ref).T@QT@(xx - xx_ref)
    lTx = QT@(xx - xx_ref)
    lTxx = QT

    return lT.squeeze(), lTx.squeeze(), lTxx.squeeze()

def Newton (xx, uu, xx_ref, uu_ref, Q, R, QT, max_iters):

    # arrays to store data

    lmbd = np.zeros((ns, T, max_iters))    # lambdas - costate seq.
    deltau = np.zeros((ni,T, max_iters))   # Du - descent direction
    dJ = np.zeros((ni,T, max_iters))       # DJ - gradient of J wrt u
    ll = np.zeros((max_iters-1))
    dl = np.zeros((ns+ni, max_iters-1))
    d2l = np.zeros((ns+ni,ns+ni, max_iters-1))
    dl_norm = np.zeros(max_iters-1)         #[for plots]
    J = np.zeros((max_iters))               # collect cost
    dJ = np.zeros((ns+ni,max_iters))
    ddJ = np.zeros((ns+ni, ns+ni, max_iters))
    direction = np.zeros((ni, T, max_iters))
    descent = np.zeros(max_iters)           # collect descent direction
    descent_arm = np.zeros(max_iters)       # collect descent direction
    Qtk = np.zeros((ns, ns, max_iters))
    Rtk = np.zeros((ns, max_iters))
    Stk = np.zeros((ns, max_iters))
    QTk = np.zeros((ns, max_iters))


    x0 = np.copy(xx_ref[:,0])

    for kk in range(max_iters-1):

        J[kk] = 0

        # calculate cost
        for tt in range(T-1):
            temp_cost = cost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
            J[kk] += temp_cost

        temp_cost = cost_f(xx[:,-1,kk], xx_ref[:,-1], QT)[0]
        J[kk] += temp_cost

        # Descent direction calculation
        lmbd_temp = cost_f(xx[:,T-1,kk], xx_ref[:,T-1], QT)[1]
        lmbd[:,T-1,kk] = lmbd_temp.squeeze()

        for tt in reversed(range(T-1)):                        # integration backward in time

            Qtk[:,kk] = cost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[2][1,1]
            Rtk[:,kk] = cost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[2][1,2]
            Stk[:,kk] = cost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[2][1,1]
            QTk[:,kk] = cost_f(xx[:,-1,kk], xx_ref[:,-1], QT)[2]

            at, bt = cost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q, R)[1:]
            fx, fu = dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]

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
                uu_temp[:,tt] = uu[:,tt,kk] + stepsize*direction[:,tt,kk]
                xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(T-1):
                temp_cost = cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
                JJ_temp += temp_cost

            temp_cost = cost_f(xx_temp[:,-1], xx_ref[:,-1], QT)[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)                              # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100*J[kk]]))       # save the cost associated to the stepsize

            if JJ_temp > J[kk]  + cc*stepsize*dJ[:,kk].T@direction:
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
                uu_temp[:,tt] = uu[:,tt,kk] + step*direction[:,tt,kk]
                xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(T-1):
                temp_cost = cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Q, R)[0]
                JJ_temp += temp_cost

            temp_cost = cost_f(xx_temp[:,-1], xx_ref[:,-1], QT)[0]
            JJ_temp += temp_cost

            costs[ii] = np.min([JJ_temp, 100*J[kk]])


        plt.figure(1)
        plt.clf()
        plt.plot(steps, costs, color='g', label='$\\ell(x^k - \\gamma*d^k$)')
        plt.plot(steps, J[kk] + dJ[:,kk].T@direction*steps, color='r', label='$\\ell(x^k) - \\gamma*\\nabla\\ell(x^k)^{\\top}d^k$')
        plt.plot(steps, J[kk] + dJ[:,kk].T@direction*steps + 1/2*direction.T@ddJ[:,:,kk]@direction*steps**2, color='b', label='$\\ell(x^k) - \\gamma*\\nabla\\ell(x^k)^{\\top}d^k - \\gamma^2 d^{k\\top}\\nabla^2\\ell(x^k) d^k$')
        plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(x^k) - \\gamma*c*\\nabla\\ell(x^k)^{\\top}d^k$')
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
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*direction[:,tt,kk]
            xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition

        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,np.linalg.norm(direction)[kk], J[kk]))

        if np.linalg.norm(direction)[kk] <= term_cond:
            max_iters = kk
            break

    return xx, uu, np.linalg.norm(direction), J