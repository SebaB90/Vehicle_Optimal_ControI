#
# Newton's Method
# Lorenzo Sforni
# Bologna, 09/10/2023
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
def cost_fcn(xx):
    """
    Input
    - Optimization variable
        x \in R^2,  x = [x1x2]
    Output
    - ll  (nonlinear) cost at x 
        l(x) = exp(x1+3*x2-0.1) + exp(x1-3x2-0.1) + exp(-x1-0.1),
    - dl gradient of ll at x, \nabla l(x)
    """
    ll = np.exp(xx[0] + 3*xx[1]-0.1) + np.exp(xx[0]-3*xx[1]-0.1) + np.exp(-xx[0]-4)
    # Compute gradient components 
    dl1 = np.exp(xx[0] + 3*xx[1] - 0.1) + np.exp(xx[0] - 3*xx[1] - 0.1) - np.exp(-xx[0] - 4)
    dl2 = 3.*np.exp(xx[0] + 3*xx[1] - 0.1) - 3*np.exp(xx[0] - 3*xx[1] - 0.1)

    d2l11 = np.exp(xx[0] + 3*xx[1] - 0.1) + np.exp(xx[0]-3*xx[1]-0.1) + np.exp(-xx[0]-4)
    d2l22 = 9*np.exp(xx[0] + 3*xx[1] - 0.1) + 9*np.exp(xx[0]-3*xx[1]-0.1)
    d2l12 = 3.*np.exp(xx[0] + 3*xx[1] - 0.1) - 3*np.exp(xx[0] - 3*xx[1] - 0.1)

    dl = np.array([dl1,dl2])
    #d2l = np.eye(n_x)
    d2l = np.array([[d2l11, d2l12], [d2l12, d2l22]])
    return ll, dl, d2l

def min_cvx_solver(xx):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/
    """
    xx = cvx.Variable(xx.shape[0])
    cost = cvx.exp(xx[0] + 3*xx[1]-0.1) + cvx.exp(xx[0]-3*xx[1]-0.1) + cvx.exp(-xx[0]-4)
  
    problem = cvx.Problem(cvx.Minimize(  
        cost
    ))
    problem.solve() # Naive implementation, no tolerances -> check gradient of x_star
    #problem.solve(solver = 'ECOS', abstol=1e-8) # Smart implementation, choose solver and tolerance
    return xx.value, problem.value


######################################################
# Main code
######################################################

max_iters = int(5e2)
stepsize = 0.01
n_x = 2           # state variable dimension

cc = 0.5

# Initialize state, cost and gradient variables (for each algorithm)
xx = np.zeros((n_x, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_x, max_iters-1))
d2l = np.zeros((n_x,n_x, max_iters-1))
dl_norm = np.zeros(max_iters-1) #[for plots]

# Set initial condition for each state variable
# plot domain x0 \in [-3.0,-0.0]
#             x1 \in [-1.5, 1.5]
xx_init = np.array([-5,1])
xx[:,0] = xx_init 

# Algorithm
for kk in range(max_iters-1):

    # Compute cost and gradient

    # Gradient Method
    ll[kk], dl[:,kk], d2l[:,:,kk] = cost_fcn(xx[:,kk])
    dl_norm[kk] = np.linalg.norm(dl[:,kk]) #[for plots]

    direction = - np.linalg.inv(d2l[:,:,kk])@dl[:,kk]

    ############################
    # Descent plot
    ############################

    steps = np.linspace(0,1,int(1e2))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

        step = steps[ii]

        xxp_temp = xx[:,kk] + step*direction   # temporary update

        costs[ii] = cost_fcn(xxp_temp)[0]

    plt.figure()

    plt.clf()
    plt.title('Descent (Second-order)')
    plt.plot(steps, costs, color='g', label='$\\ell(x^k + \\gamma*d^k$)')
    plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps, color='r', label='$\\ell(x^k) + \\gamma*\\nabla\\ell(x^k)^{\\top}d^k$')
    plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps + 1/2*direction.T@d2l[:,:,kk]@direction*steps**2, color='b', label='$\\ell(x^k) + \\gamma*\\nabla\\ell(x^k)^{\\top}d^k + \\gamma^2 d^{k\\top}\\nabla^2\\ell(x^k) d^k$')
    plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(x^k) + \\gamma*c*\\nabla\\ell(x^k)^{\\top}d^k$')
    plt.grid()
    plt.legend()

    plt.show()

    ############################
    ############################


    xx[:,kk+1] = xx[:,kk] + stepsize * direction

    print('Iter {}\tCost:\tGradient = {}'.format(kk,ll[kk]))

# Compute optimal solution via cvx solver [plot]
xx_star, ll_star = min_cvx_solver(xx_init)
print('ll_star = {}'.format(ll_star), '\tx_star = {}'.format(xx_star))
print('dl_star = {}'.format(cost_fcn(xx_star)[1]))

######################################################
# Plots
######################################################


plt.figure('descent direction')
plt.plot(np.arange(max_iters-1), dl_norm)
plt.xlabel('$k$')
plt.ylabel('||$d\ell(x^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.figure('cost error')
plt.plot(np.arange(max_iters-1), abs(ll-ll_star))
plt.xlabel('$k$')
plt.ylabel('$||\ell(x^{k})-\ell(x^{*})||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure()
plt.rcParams.update({'font.size': 12})
domain_x = np.arange(-3,3,0.1)
domain_y = np.arange(-3,3,0.1)
domain_x, domain_y = np.meshgrid(domain_x, domain_y)
cost_on_domain = np.zeros(domain_x.shape)

for ii in range(domain_x.shape[0]):
    for jj in range(domain_x.shape[1]):
        cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_x[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

ax = plt.axes(projection='3d')
ax.plot_surface(domain_x, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.8)
ax.plot3D(xx[0,:max_iters-1], xx[1,:max_iters-1], ll, color = 'tab:blue')
ax.scatter3D(xx[0,:max_iters-1], xx[1,:max_iters-1], ll, color = 'tab:blue')
ax.set_xlabel('$x^k_0$')
ax.set_ylabel('$x^k_1$')
ax.set_zlabel('$\ell(x^k)$')

plt.show()

