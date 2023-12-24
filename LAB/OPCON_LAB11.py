#
# Gradient Method - Scalar
# Lorenzo Sforni
# Bologna, 16/10/2023
#

import numpy as np
import matplotlib.pyplot as plt

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Set some plot parameters
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

######################################################
# Functions
######################################################
def cost_fcn(xx):
    """
    Input
    - Optimization variable
        x \in R 
    Output
    - ll cost at x 
        l(x) = x^2,
    - dl gradient of ll at x, \nabla l(x) = x
    """

    ll = xx**2 
    dl = 2*xx
    return ll, dl


######################################################
# Main code
######################################################

max_iters = 100
stepsize = 0.2

# Initialize state, cost and gradient variables
xx = np.zeros(max_iters)
ll = np.zeros(max_iters)
dl = np.zeros(max_iters)

# Set initial condition for algorithms
xx_init = -5
xx[0] = xx_init 


# Algorithm
for kk in range(max_iters-1):

    # Compute cost and gradient
    ll[kk], dl[kk] = cost_fcn(xx[kk]) 
    
    # Select the direction
    direction = - dl[kk]

    ############################
    # Descent plot
    ############################

    steps = np.linspace(0,1,int(1e2))
    costs = np.zeros(len(steps))

    cc = 0.5

    for ii in range(len(steps)):

        step = steps[ii]

        xxp_temp = xx[kk] + step*direction   # temporary update

        costs[ii] = cost_fcn(xxp_temp)[0]

        
    # Update the solution
    xx[kk+1] = xx[kk] + stepsize * direction

    print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,xx[kk+1]))

plt.figure()

plt.clf()
plt.title('Descent')
plt.plot(steps, costs, color='g', label='$\\ell(x^k + stepsize*d^k$)')
plt.plot(steps, ll[kk] + dl[kk].T*direction*steps, color='r', label='$\\ell(x^k) + stepsize*\\nabla\\ell(x^k)^{\\top}d^k$')
plt.plot(steps, ll[kk] + cc*dl[kk].T*direction*steps, color='g', linestyle='dashed', label='$\\ell(x^k) + stepsize*c*\\nabla\\ell(x^k)^{\\top}d^k$')
plt.grid()
plt.legend()

plt.show()

######################################################
# Plots
######################################################

# Evaluate function cost over its domain [plots]
domain = np.arange(-7,7,0.1)
cost_on_domain = np.zeros(domain.shape)

for ii, xx_dom in enumerate(domain):
    cost_on_domain[ii] = cost_fcn(xx_dom)[0]

plt.figure()
plt.plot(domain, cost_on_domain, linestyle = 'dashed', color='tab:blue')
plt.plot(xx, ll, color='tab:orange')
plt.scatter(xx, ll, color='tab:orange', marker='o')
plt.xlabel('$x^k$')
plt.ylabel('$\ell(x^k)$')
plt.xlim(min(domain), max(domain))
plt.ylim(min(cost_on_domain), max(cost_on_domain))
plt.legend(['$\ell_{domain}$','$\ell(x^k)$'])
plt.grid()
plt.show(block=False)

plt.figure()
plt.plot(np.arange(max_iters-1), abs(xx[1:max_iters] - xx[0:max_iters-1]))
plt.xlabel('$k$')
plt.ylabel('$||x^{k+1}-x^{k}||$')
plt.yscale('log')
plt.grid()
plt.show()