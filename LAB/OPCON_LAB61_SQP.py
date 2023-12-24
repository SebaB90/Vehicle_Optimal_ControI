#
# SQP method for constrained optimization
# Lorenzo Sforni
# Bologna, 26/10/2023
#


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

##############################
# SQP solver
##############################

def SQP_solver(xk, dl, hk, dh, Bk):
  """
      SQP solver using CVXPY
    
      min_{x \in \R^n} dl^T (x - xk) + 0.5*(x-xk)^T B (x-xk)
      subj.to h + dh^T (x-xk) = 0
  """

  x = cvx.Variable(xk.shape)

  cost = dl.T@(x - xk) + 0.5*cvx.quad_form(x - xk, Bk)
  constraint = [hk + dh.T@(x-xk) == 0]

  problem = cvx.Problem(cvx.Minimize(cost), constraint)
  problem.solve()

  x_star = x.value
  lambda_QP = constraint[0].dual_value


  return x_star, lambda_QP

  
##############################
# Problem
##############################

# Parameters
r = 1; x0 = 0; y0 = 2 # constraint parameters

def cost_fcn(xx):
    """
    Input
    - Optimization variable
        x \in R^2,  x = [x1x2]
    Output
    - ll cost at x 
        l(x)    = 0.5 x\T Q x + q\T x
    - dl gradient of ll at x, \nabla l(x)
        dl(x)   = Q x + q
    - d2l hessian of ll at x, \nabla^2 l(x)
        d2l(x)  = Q
    """

    Q = np.array([[2,1],[1,4]])
    # Q = np.array([[0.1,0.1],[0.1,0.4]])
    q = np.array([2,0])

    ll = 0.5*xx.T@Q@xx + q@xx
    dl = Q@xx + q
    d2l = Q


    return ll, dl, d2l

def equality_constraint_fcn(xx):
    """
    Equality constraint
    - h(x) = || x - x_c ||^2 - r^2 = 0
    - dh(x) = 2(x - x_c)
    """

    xxc = np.array([x0, y0])

    hh = (xx[0] - xxc[0])**2 + (xx[1] - xxc[1])**2 - r**2
    dh = 2*np.array([xx[0] - xxc[0], xx[1] - xxc[1]])

    return hh, dh

def merit_fcn(xk, lmb, deltax):
    """
     Merit function
     - M1(x) = l(x) + s || g(x) ||_1
     - Directional derivative
        D_Dx M1(x) = dl(x)^T Dx - s || g(x) ||_1
        s > || lmd ||_{inf}
    """

    # Compute cost and gradient
    lk, dlk = cost_fcn(xk)[:2]

    # Compute constraint and gradient
    hk = equality_constraint_fcn(xk)[0] 

    s = 100*np.ceil(np.linalg.norm([lmb], np.inf))

    M1 = lk + s*np.linalg.norm([hk], 1)

    DM1 = dlk.T@deltax - s*np.linalg.norm([hk], 1)

    return M1, DM1.squeeze()


##############################
# Optimization
##############################


max_iters = int(1e2)
n_x = 2           # state variable dimension

# Choose methods

methods = ['SQP', 'Newton']
method = 'Newton'

# ARMIJO PARAMETERS

stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_maxiters = 10 # number of Armijo iterations

# Initialize state, cost and gradient variables (for each algorithm)
xx = np.zeros((n_x, max_iters))
lmbd = np.zeros((max_iters))
ll_mult = np.zeros((n_x, max_iters))  # multipliers

ll = np.zeros((max_iters-1))
dl = np.zeros((n_x, max_iters-1))
hh = np.zeros((max_iters-1))
dh = np.zeros((n_x, max_iters-1))
BB = np.zeros((n_x, n_x, max_iters-1))

descent_norm = np.zeros(max_iters-1) #[for plots]
lagrangian_norm = np.zeros(max_iters-1) #[for plots]


x_init = [0,2.1]    # initial condition s.t. regularization is required
x_init = [-2,2]
xx[:,0] = x_init 

stepsize = stepsize_0

# Algorithm
for kk in range(max_iters-1):

    # Compute cost and gradient
    ll[kk], dl[:,kk], d2l= cost_fcn(xx[:,kk]) 

    # Compute constraint and gradient
    hh[kk], dh[:,kk] = equality_constraint_fcn(xx[:,kk]) 
    
    #BB[:,:,kk] = d2l + 2*lmbd[kk]*np.eye(n_x) # Exact Hessian
    BB[:,:,kk] = d2l                           # Gauss-Newton Approximation

    # check positive definiteness of BB
    print(np.linalg.eigvals(BB[:,:,kk]))
    if np.any(np.linalg.eigvals(BB[:,:,kk]) <= 0):
        print('Hessian not positive definite')
        break

    if method == methods[0]:

        # SQP solver
        xqp, lmb_qp = SQP_solver(xx[:,kk], dl[:,kk], hh[kk], dh[:,kk], BB[:,:,kk])
        #
        # compute the direction and multiplier
        direction = (xqp - xx[:,kk])
        multiplier = lmb_qp

    elif method == methods[1]:

        # build block matrix
        W = np.block([[BB[:,:,kk],dh[:,kk][:,None]], [dh[:,kk][:,None].T, 0]])
        #
        # compute newton direction solving linear system
        newt_dir = np.linalg.solve(W, np.block([-dl[:,kk], -hh[kk]]))
        #
        # compute the direction and multiplier
        direction = newt_dir[:-1]
        multiplier = newt_dir[-1]


    descent_norm[kk] = np.linalg.norm(direction) #[for plots]
    lagrangian_norm[kk] = np.linalg.norm(dl[:,kk] + dh[:,kk]*multiplier, np.inf) #[for plots]

    ############################
    # Armijo stepsize selection
    ############################

    stepsizes = []  # list of stepsizes
    merits_armijo = []

    stepsize = stepsize_0

    M1k,DM1k = merit_fcn(xx[:,kk], multiplier, direction)


    for ii in range(armijo_maxiters):
        
        xxp_temp = xx[:,kk] + stepsize*direction   # temporary update

        MM_temp = merit_fcn(xxp_temp, multiplier, direction)[0]

        stepsizes.append(stepsize)      # save the stepsize
        merits_armijo.append(MM_temp)    # save the cost associated to the stepsize

        if MM_temp > M1k + cc*stepsize*DM1k:
            
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {}'.format(stepsize))
            break

    ############################
    # Descent plot
    ############################

    steps = np.linspace(0,stepsize_0,int(1e2))
    MM = np.zeros(len(steps))

    M1k,DM1k = merit_fcn(xx[:,kk], multiplier, direction)

    for ii in range(len(steps)):

        step = steps[ii]

        xxp_temp = xx[:,kk] + step*direction   # temporary update

        MM[ii] = merit_fcn(xxp_temp, multiplier, direction)[0]


    ############################
    # Plot
    #
    plt.figure(1)
    plt.clf()
    #
    plt.title('Descent (Merit function)')
    plt.plot(steps, MM, color='g', label='$M_1(x^k + \\gamma d^k)$')
    plt.plot(steps, M1k + steps*DM1k, color='r', \
             label='$M_1(x^k) + \\gamma\mathrm{D}_{d^k} M_1(x^k, \lambda^k)$')
    plt.plot(steps, M1k + cc*steps*DM1k, color='g', linestyle='dashed', \
             label='$M_1(x^k) + c\\gamma \mathrm{D}_{d^k} M_1(x^k, \lambda^k)$')
    #
    plt.scatter(stepsizes, merits_armijo, marker='*') # plot the tested stepsize
    #
    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()
    #
    plt.show()
    ############################
    
    ############################
    # Update solution and multiplier
    #
    xx[:,kk+1] = xx[:,kk] + stepsize*direction
    lmbd[kk+1] = (1 - stepsize)*lmbd[kk] + stepsize*multiplier



    print(descent_norm[kk])
    print(f'LagNorm = {np.linalg.norm(dl[:,kk] + dh[:,kk]*lmbd[kk], np.inf)}')
    print(f'ConstrNorm = {np.linalg.norm(hh[kk])}')
    print(lmbd[kk+1])
    if kk%1e2 == 0:
        print('ll_{} = {}'.format(kk,ll[kk]))

    if np.linalg.norm(direction) <= 1e-10:
    
        max_iters = kk+1

        break


# Compute optimal solution via cvx solver [plot]
print('xx_max = {}'.format(xx[:,max_iters-1]))

######################################################
# Plots
######################################################


if 1:

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters-1), descent_norm[:max_iters-1])
    plt.xlabel('$k$')
    plt.ylabel('||$d\ell(x^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    plt.figure('lagrangian norm')
    plt.plot(np.arange(max_iters-1), lagrangian_norm[:max_iters-1])
    plt.xlabel('$k$')
    plt.ylabel('$||\\nabla_x\mathcal{L}(x^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

plt.figure()
plt.rcParams.update({'font.size': 12})

domain_x = np.arange(-10,10,0.1)
domain_y = np.arange(-10,10,0.1)
domain_x, domain_y = np.meshgrid(domain_x, domain_y)
cost_on_domain = np.zeros(domain_x.shape)

# draw constraint
t = np.linspace(0, 2*np.pi, 100)
x1_t = x0 + r*np.cos(t)
x2_t = y0 + r*np.sin(t)

# evaluate cost on x_t, y_t
ft = np.zeros(x1_t.shape)
for ii in range(x1_t.shape[0]):
    ft[ii] = cost_fcn(np.array((x1_t[ii],x2_t[ii])))[0]


for ii in range(domain_x.shape[0]):
    for jj in range(domain_x.shape[1]):
        cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_x[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

ax = plt.axes(projection='3d')
ax.plot_surface(domain_x, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.4)
ax.plot3D(xx[0,:max_iters-1], xx[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange')
ax.scatter3D(xx[0,:max_iters-1], xx[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange', s=50)
ax.plot3D(x1_t, x2_t, ft, 'r', linewidth=2)
ax.set_xlabel('$x^k_0$')
ax.set_ylabel('$x^k_1$')
ax.set_zlabel('$\ell(x^k)$')

plt.show()