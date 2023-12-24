#
# Gradient Method - QP
# Lorenzo Sforni
# Bologna, 18/10/2022
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

######################################################
# Functions
######################################################
def cost_fcn(xx, QQ, qq):
    """
    Input
    - Optimization variable
        x \in R^n,  x = [x1 x2 ... xn]
    Output
    - ll  cost at x 
        l(x) = 0.5 * x.T*Q*x + q.T*x,
    - dl gradient of ll at x, \nabla l(x)
    """
    ll  = 0.5 * xx.T @ QQ @ xx + qq.T @ xx
    dl = 0.5 * (QQ+QQ.T) @ xx + qq
    return ll, dl

def min_cvx_solver(QQ, qq):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/

    Obtain optimal solution for unconstrained QP
        min_{x} 1/2 x^T Q x + q^T x
    """
    xx = cvx.Variable(qq.shape)
    cost = 0.5* cvx.quad_form(xx,QQ) + qq.T @ xx
    problem = cvx.Problem(cvx.Minimize(  
        cost
    ))
    problem.solve()
    return xx.value, problem.value


######################################################
# Main code
######################################################

np.random.seed(10)

max_iters = int(1e3)
stepsize_0 = 1
n_x = 5             # state variable dimension


beta = 0.7
cc =0.5

# Set problem parameters
QQ = np.diag(np.random.rand(n_x))
qq = np.random.rand(n_x)


# Initialize state, cost and gradient variables (for each algorithm)
xx = np.zeros((n_x, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_x, max_iters))
dl_norm = np.zeros(max_iters-1) #[for plots]

# Set initial condition for algorithms
xx_init = np.random.rand(n_x)
xx[:,0] = xx_init 

# Algorithm
for kk in range(max_iters-1):

    # compute cost and gradient
    ll[kk], dl[:,kk] = cost_fcn(xx[:,kk], QQ, qq) 
    
    dl_norm[kk] = np.linalg.norm(dl[:,kk]) #[for plots]

    # Select the direction
    direction = - dl[:,kk]

    ############################
    # Armijo stepsize selection
    ############################

    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0

    for ii in range(10):
        
        xxp_temp = xx[:,kk] + stepsize*direction   # temporary update

        ll_temp = cost_fcn(xxp_temp, QQ, qq)[0]

        stepsizes.append(stepsize)      # save the stepsize
        costs_armijo.append(ll_temp)    # save the cost associated to the stepsize

        if ll_temp > ll[kk] + cc*stepsize*dl[:,kk].T@direction:
            
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {}'.format(stepsize))
            break


    ############################
    # Descent plot
    ############################

    steps = np.linspace(0,1,int(1e3))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

        step = steps[ii]

        xxp_temp = xx[:,kk] + step*direction   # temporary update

        costs[ii] = cost_fcn(xxp_temp, QQ, qq)[0]

    plt.figure()

    plt.clf()
    plt.title('Descent (second order)')
    plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps, color='r', label='$\\ell(x^k) + stepsize*\\nabla\\ell(x^k)^{\\top}d^k$')
    plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(x^k) + stepsize*c*\\nabla\\ell(x^k)^{\\top}d^k$')
    plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps + 1/2*direction.T@QQ@direction*steps**2, color='b', label='$\\ell(x^k) + \\gamma*\\nabla\\ell(x^k)^{\\top}d^k + \\gamma^2 d^{k\\top}\\nabla^2\\ell(x^k) d^k$')
    plt.plot(steps, costs, color='g', label='$\\ell(x^k + stepsize*d^k$)')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

    plt.grid()
    plt.legend()

    plt.show()

    ############################
    ############################

    xx[:,kk+1] = xx[:,kk] + stepsize * direction
    print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,xx[:,kk+1]))

    if np.linalg.norm(direction) <= 1e-4:
        
        max_iters = kk+1

        break

# Compute optimal solution via cvx solver [plot]
xx_star, ll_star = min_cvx_solver(QQ, qq)
print('ll_star = {}'.format(ll_star), '\tx_star = {}'.format(xx_star))

######################################################
# Plots
######################################################

plt.figure()
plt.plot(np.arange(max_iters-1), dl_norm[:max_iters-1])
plt.xlabel('$k$')
plt.ylabel('||$d\ell(x^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure('cost error')
plt.plot(np.arange(max_iters-1), abs(ll[:max_iters-1]-ll_star))
plt.xlabel('$k$')
plt.ylabel('$||\ell(x^{k})-\ell(x^{*})||$')
plt.yscale('log')
plt.grid()
plt.show()