#
# Gradient Method - QP
# Lorenzo Sforni
# Bologna, 18/10/2022
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

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

def min_cvx_solver(QQ, qq, AA, bb):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/

    Obtain optimal solution for constrained QP

        min_{x} 1/2 x^T Q x + q^T x
        s.t.    Ax - b <= 0

    """
    xx = cvx.Variable(qq.shape)

    # Cost function
    cost = 0.5* cvx.quad_form(xx,QQ) + qq.T @ xx

    # Constraint Ax <= b
    constraint = [AA@xx <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    problem.solve()
    return xx.value, problem.value

def projection(xx, AA, bb):
    """
        Projection using CVXPY

        min_{z \in \R^n} || z - x||^2
        s.t. Az - b <= 0
    
    """

    zz = cvx.Variable(xx.shape)

    cost = cvx.norm(zz - xx, 2)
    constraint = [AA@zz <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    problem.solve()

    return zz.value


######################################################
# Main code
######################################################

np.random.seed(10)

max_iters = int(5e2)
stepsize = 1e-1
n_x = 5 # state variable dimension

n_p = 4 # constraints dimension            

# Set problem parameters

# Cost

QQ = np.diag(np.random.rand(n_x))
qq = np.random.rand(n_x)

# Constraint

AA = np.random.rand(n_p,n_x)
bb = 3*np.random.rand(n_p)

# Initialize state, cost and gradient variables (for each algorithm)
xx = np.zeros((n_x, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_x, max_iters))

# Set initial condition for algorithms
# Find feasible initial condition
xx_init = np.random.rand(n_x)
while not np.all(AA@xx_init - bb < 0):
    xx_init = np.random.rand(n_x)
    print(AA@xx_init - bb)


xx[:,0] = xx_init 

# Algorithm
for kk in range(max_iters-1):

    # compute cost and gradient
    ll[kk], dl[:,kk] = cost_fcn(xx[:,kk], QQ, qq) 

    # Select the direction
    direction = - dl[:,kk]

    # Update the solution
    xx_temp = xx[:,kk] + stepsize * direction

    # Projection step
    xx[:,kk+1] = projection(xx_temp, AA, bb)

    print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,xx[:,kk+1]))

# Compute optimal solution via cvx solver [plot]
xx_star, ll_star = min_cvx_solver(QQ, qq, AA, bb)
print('ll_star = {}'.format(ll_star), '\tx_star = {}'.format(xx_star))

######################################################
# Plots
######################################################

plt.figure()
plt.plot(np.arange(max_iters-1), ll)
plt.plot(np.arange(max_iters-1), ll_star*np.ones(max_iters-1), linestyle = 'dashed')
plt.xlabel('$k$')
plt.ylabel('$\ell(x^k)$')
plt.legend(['$\ell(x^{k})$','$\ell(x^{*})$'])
plt.grid()


plt.figure()
plt.plot(np.arange(max_iters-1), abs(ll-ll_star))
plt.title('cost error')
plt.xlabel('$k$')
plt.ylabel('$||\ell(x^{k})-\ell(x^{*})||$')
plt.yscale('log')
plt.grid()


plt.show()