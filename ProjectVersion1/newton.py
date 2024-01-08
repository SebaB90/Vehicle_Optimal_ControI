#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#
import sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import Dynamics as dyn
import Costs as cst

#define params
ns = dyn.ns  # number of states
ni = dyn.ni  # number of inputs
dt = dyn.dt  # sample time

TT = dyn.TT  # Final time in seconds
tf = dyn.tf   # Number of discrete-time samples
TT_mid = dyn.TT_mid

term_cond = 1e-6        #terminal condition

# ARMIJO PARAMETERS
c = 0.5
beta = 0.7
armijo_maxiters = 20    # number of Armijo iterations
stepsize_0 = 1          # initial stepsize
armijo_plt = False

# Import the cost matrices from costs
Qt = cst.QQt
Rt = cst.RRt
QT = cst.QQT

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None, ccin = None):
    
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

    '''

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
    cc = ccin

    qqf = qqfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0
    
    PP[:,:,-1] = QQf
    pp[:,-1] = qqf

    # Evaluate KK and PP
    
    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]
        cct = cc[:,tt][:,None]

        # Evaluate K
        #if pptp[0,0] != pptp[0,0]:
         #  print('\n\nqqt + AAt.T@pptp + AAt.T@PPtp@cct + KKt.T@(RRt + BBt.T@PPtp@BBt)@sigma_t',qqt + AAt.T@pptp + AAt.T@PPtp@cct + KKt.T@(RRt + BBt.T@PPtp@BBt)@sigma_t,
          #       '\n\qqt',qqt,'\n\AAt',AAt,'\n\pptp',pptp,'\n\KKt',KKt,'\n\RRt',RRt,'\n\sigma_t',sigma_t)
         #  sys.exit()
        
        if pptp[0,0] != pptp[0,0]:
            print('\n\pptp0')
            sys.exit()
        if pptp[1,0] != pptp[1,0]:
            print('\n\pptp1')
            sys.exit()


        KKt =  -np.linalg.inv((RRt+BBt.T@PPtp@BBt)) @(SSt+BBt.T@PPtp@AAt)
        sigma_t = -np.linalg.inv((RRt+BBt.T@PPtp@BBt)) @(rrt+BBt.T@pptp+BBt.T@PPtp@cct)


        if sigma_t[0,0] != sigma_t[0,0]:
            print('\n\nrrt+BBt.T@pptp+BBt.T@PPtp@cct',rrt+BBt.T@pptp+BBt.T@PPtp@cct,'\n\nrrt',rrt,'\n\npptp',pptp,'\n\ncct',cct)
            sys.exit()
           
        KK[:,:,tt] = KKt
        sigma[:,tt] = sigma_t.squeeze()

        # Solve Riccati Equation

        ppt = qqt + AAt.T@pptp + AAt.T@PPtp@cct + KKt.T@(RRt + BBt.T@PPtp@BBt)@sigma_t
        PPt = QQt + AAt.T@PPtp@AAt - KKt.T@(RRt + BBt.T@PPtp@BBt)@KKt

        if ppt[0,0] == float('-inf'):
          print('\n\nqqt + AAt.T@pptp + AAt.T@PPtp@cct + KKt.T@(RRt + BBt.T@PPtp@BBt)@sigma_t',qqt + AAt.T@pptp + AAt.T@PPtp@cct + KKt.T@(RRt + BBt.T@PPtp@BBt)@sigma_t,
               '\n\nqqt',qqt,'\n\nAAt',AAt,'\n\npptp',pptp,'\n\nKKt',KKt,'\n\nRRt',RRt,'\n\nsigma_t',sigma_t,'\n\nPPtp',PPtp)
          sys.exit()


        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()
       

    for tt in range(TT - 1):
        # Trajectory
        uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
        xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

        xx[:,tt+1] = xx_p
    '''
    
    return xx, uu, KK, sigma


def Newton (xx, uu, xx_ref, uu_ref, x0, max_iters):

    # arrays to store data
    A = np.zeros((ns, ns, TT))
    B = np.zeros((ns, ni, TT))
    d1l = np.zeros((ns, TT))
    d2l = np.zeros((ni, TT))

    cc = np.zeros((ns,TT))
    xx0 = np.zeros((ns,))

    Qtilda = np.zeros((ns, ns, TT))
    Rtilda = np.zeros((ni, ni, TT))
    Stilda = np.zeros((ni, ns, TT))
    
    lmbd = np.zeros((ns, TT, max_iters+1))    # lambdas - costate seq.

    dJ = np.zeros((ni,TT, max_iters+1))       # DJ - gradient of J wrt u
    J = np.zeros(max_iters+1)                 # collect cost
    descent = np.zeros(max_iters+1)           # collect descent direction
    descent_arm = np.zeros(max_iters+1)       # collect descent direction     

    Dx = np.zeros((ns, TT, max_iters+1))
    Du = np.zeros((ni, TT, max_iters+1))

    ################################################################################################################
    
    for kk in range(max_iters):
        J[kk] = 0

        # Parameters evaluation

        for tt in range(TT-1):
            temp_cost= cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
            J[kk] += temp_cost
            fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]
            A[:,:,tt] = fx.T
            B[:,:,tt] = fu.T

        temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
        J[kk] += temp_cost

        # Descent direction calculation
        lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()


        for tt in reversed(range(TT-1)):                        # integration backward in time

            a, b = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:3]           

            d1l[:,tt] = a.squeeze()
            d2l[:,tt] = b.squeeze()

            lmbd_temp = A[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + a       # costate equation
            dJ_temp = B[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + b         # gradient of J wrt u 
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()


        # Matrices evaluation
        for tt in range(TT-1):
            Qtilda[:,:,tt], Rtilda[:,:,tt], Stilda[:,:,tt] = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[3:]
        d1lT, QTilda = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[1:3]
        
        Dx[:,:,kk], Du[:,:,kk], KK, sigma = ltv_LQR(A, B, Qtilda, Rtilda, Stilda, QTilda, TT, xx0, d1l, d2l, d1lT.squeeze(), cc)


        for tt in reversed(range(TT-1)): 
            descent[kk] += Du[:,tt,kk].T@Du[:,tt,kk] 
            descent_arm[kk] += dJ[:,tt,kk].T@Du[:,tt,kk] 

        # Stepsize selection - ARMIJO
        stepsizes = []  # list of stepsizes
        costs_armijo = []

        stepsize = stepsize_0

        if 1:
            for ii in range(armijo_maxiters):

                # temp solution update

                xx_temp = np.zeros((ns,TT))
                uu_temp = np.zeros((ni,TT))
                descent_arm[kk] = 0

                xx_temp[:,0] = x0

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + stepsize*Du[:,tt,kk]
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

                JJ_temp = 0

                for tt in range(TT-1):
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost

                temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                JJ_temp += temp_cost

                stepsizes.append(stepsize)                              # save the stepsize
                costs_armijo.append(np.min([JJ_temp, 100*J[kk]]))       # save the cost associated to the stepsize

                if JJ_temp > J[kk] + c*stepsize*descent_arm[kk]:
                    # update the stepsize
                    stepsize = beta*stepsize

                else:
                    print('Armijo stepsize = {:.3e}'.format(stepsize))
                    break
        
        # Armijo plot
        if armijo_plt:
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                step = steps[ii]
                # temp solution update

                xx_temp = np.zeros((ns,TT))
                uu_temp = np.zeros((ni,TT))

                xx_temp[:,0] = x0

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + step*Du[:,tt,kk]
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

                # temp cost calculation
                JJ_temp = 0

                for tt in range(TT-1):
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost

                temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                JJ_temp += temp_cost

                costs[ii] = np.min([JJ_temp, 100*J[kk]])

            plt.figure(1)
            plt.clf()
            plt.plot(steps, costs, color='g', label='$\\ell(x^k - \\gamma*d^k$)')
            plt.plot(steps, J[kk] + descent_arm[kk]*steps, color='r', label='$\\ell(x^k) - \\gamma*\\nabla\\ell(x^k)^{\\top}d^k$')
            # plt.plot(steps, J[kk] + dJ[:,kk].T@Du[:,:,kk][:,kk]*steps + 1/2*Du[:,:,kk][:,kk].T@ddJ[:,:,kk]@Du[:,:,kk][:,kk]*steps**2, color='b', label='$\\ell(x^k) - \\gamma*\\nabla\\ell(x^k)^{\\top}d^k - \\gamma^2 d^{k\\top}\\nabla^2\\ell(x^k) d^k$')
            plt.plot(steps, J[kk] + c*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$\\ell(x^k) - \\gamma*c*\\nabla\\ell(x^k)^{\\top}d^k$')
            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()
            plt.show()

        # Update the current solution

        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = x0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*Du[:,tt,kk]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]


        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition
    
        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk+1,descent[kk], J[kk]))

        if descent[kk] <= term_cond:
            break

    return xx, uu, descent, J, kk


