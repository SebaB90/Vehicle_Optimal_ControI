#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


##################################
##### TASK 0: DISCRETIZATION #####
##################################


#define params

dt = 1e-3   #sample time
dx = 1e-3   #infinitesimal increment
du = 1e-3   #infinitesimal increment
ns = 6      #number of states
ni = 2      #number of inputs
max_iters = int(3e2)    #maximum number of iterations for Newton's method


# ARMIJO PARAMETERS
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

m = 1480    #Kg
Iz = 1950   #Kg*m^2
a = 1.421   #m
b = 1.029   #m
mi = 1      #nodim
g = 9.81    #m/s^2

def dynamics (x, u):

    x_plus = np.zeros((ns,))
    Beta = [u[0] - (x[3]*np.sin(x[4]) + a*x[5])/(x[3]*np.cos(x[4])), - (x[3]*np.sin(x[4]) - b*x[5])/(x[3]*np.cos(x[4]))]    # Beta = [Beta_f, Beta_r]
    Fz = [m*g*b/(a+b), m*g*a/(a+b)]             # Fz = [F_zf, F_zr]
    Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]   # Fy = [F_yf, F_yr]

    x_plus[0] = x[0] + dt*(x[3] * np.cos(x[4]) * np.cos(x[2]) - x[3] * np.sin(x[4]) * np.sin(x[2]))                                 # X dot
    x_plus[1] = x[1] + dt*(x[3] * np.cos(x[4]) * np.sin(x[2]) + x[3] * np.sin(x[4]) * np.cos(x[2]))                                 # Y dot
    x_plus[2] = x[2] + dt*x[5]                                                                                                      # Psi dot
    x_plus[3] = x[3] + dt*((Fy[1] * np.sin(x[4]) + u[1] * np.cos(x[4] - u[0]) + Fy[0] * np.sin(x[4] - u[0]))/m)                     # V dot
    x_plus[4] = x[4] + dt*((Fy[1] * np.cos(x[4]) + Fy[0] * np.cos(x[4] - u[0]) - u[1] * np.sin(x[4] - u[0]))/(m * x[3]) - x[5])     # Beta dot
    x_plus[5] = x[5] + dt*(((u[1] * np.sin(u[0]) + Fy[0] * np.cos(u[0])) * a - Fy[1] * b)/Iz)                                       # Psi dot dot

    fx = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [dt*(-x[3] * np.cos(x[4]) * np.sin(x[2]) - x[3] * np.sin(x[4]) * np.cos(x[2])), dt*(x[3] * np.cos(x[4]) * np.cos(x[2]) - x[3] * np.sin(x[4]) * np.sin(x[2])), 1, 0, 0, 0],
                  [dt*(np.cos(x[4]) * np.cos(x[2]) - np.sin(x[4]) * np.sin(x[2])), dt*(np.cos(x[4]) * np.sin(x[2]) + np.sin(x[4]) * np.cos(x[2])), 0, 1, dt*((Fy[1] * np.cos(x[4]) + Fy[0] * np.cos(x[4] - u[0]) - u[1] * np.sin(x[4] - u[0]))*(-1/(m*(x[3]**2)))), 0],
                  [dt*(-x[3] * np.sin(x[4]) * np.cos(x[2]) - x[3] * np.cos(x[4]) * np.sin(x[2])), dt*(-x[3] * np.sin(x[4]) * np.sin(x[2]) + x[3] * np.cos(x[4]) * np.cos(x[2])), 0, dt*((Fy[1] * np.cos(x[4]) - u[1] * np.sin(x[4] - u[0]) + Fy[0] * np.cos(x[4] - u[0]))/m), 1 + dt*((-Fy[1] * np.sin(x[4]) - Fy[0] * np.sin(x[4] - u[0]) - u[1] * np.cos(x[4] - u[0]))/(m * x[3])), 0],
                  [0, 0, dt, 0, -dt, 1]])
    
    fu = np.array([[0, 0, 0, dt*(u[1] * np.sin(x[4] - u[0]) - Fy[0] * np.cos(x[4] - u[0]))/m, dt*(Fy[0] * np.sin(x[4] - u[0]) + u[1] * np.cos(x[4] - u[0]))/(m * x[3]),  dt*((u[1] * np.cos(u[0]) - Fy[0] * np.sin(u[0])) *a/Iz)],
                  [0, 0, 0, dt*np.cos(x[4] - u[0])/m, dt*(- np.sin(x[4] - u[0]))/(m * x[3]), dt * np.sin(u[0]) * a/Iz]])

    return x_plus, fx, fu

# defining x and u
u = np.array([0.25, 20])
x = np.array([0, 0, 0, 1, 0, 0])

x_plus, fx, fu = dynamics(x, u)

A = fx.T
B = fu.T

# OPEN LOOP TEST to check if the dynamics do what expected ---------------------------------------------------

x_traj = [np.copy(x[0])]
y_traj = [np.copy(x[1])]
traj = np.copy(x)

total_time = 100                     # Adjust the total simulation time as needed
num_steps = int(total_time / dt)

for _ in range(num_steps):
    traj, _, _ = dynamics(traj, u)
    x_traj.append(traj[0])
    y_traj.append(traj[1])

# Plotting the trajectory
plt.plot(x_traj, y_traj, label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Checking derivatives

# CHECK IF THE DERIVATIVES ARE CORRECT ----------------------------------------------------------------------

xdx = np.zeros((ns,))
ddx = np.zeros((ns,))
udu = np.zeros((ni,))
ddu = np.zeros((ni,))

for i in range (0,ns):
    ddx[i] = dx

for k in range (0,ni):
    ddu[k] = du

xdx = x + ddx
xx_plus = dynamics(xdx, u)[0]
diff_x = xx_plus - x_plus
check_x = diff_x - np.dot(A,ddx)

udu = u + ddu
xx_plus = dynamics(x, udu)[0]    
diff_u = xx_plus - x_plus      
check_u = diff_u - np.dot(B,ddu)

print ('error in derivatives of x is:', check_x)
print ('error in derivatives of u is:', check_u)


#########################################
##### TASK 1: TRAJECTORY GENERATION #####
#########################################

# We have to find the eqilibria for the system, a way to do that is to use the cornering equilibria, those associated to the systems with Betadot, Vdot and Psidotdot = 0
# Once I have set them I can focus on the last three equation, then imposing Veq and PsidotEq (I can think of this also as Veq/R with R a certain imposed radious) we obtain Betaeq, Fxeq and Deltaeq, in alternative I can set Veq and Betaeq and as concequence find the other eqilibrium values.
# The associated x and y trajectory can then be obtained by forward integration of the dynamics with the values we just found.
# For vehicles these trajectories are called corering eqilibria, in which I have circles with some radious and some Veq.

# Evaluate the EQUILIBRIUM  ----------------------------------------------------------------------------------

# imposed parameters  

eq = np.zeros((5, 2))                                  

# calculation of the other parameters
def equations(vars):
    x5, u0, u1 = vars
    Beta = [u0 - (x3*np.sin(x4) + a*x5)/(x3*np.cos(x4)), - (x3*np.sin(x4) - b*x5)/(x3*np.cos(x4))]              # Beta = [Beta_f, Beta_r]
    Fz = [m*g*b/(a+b), m*g*a/(a+b)]                                                                             # Fz = [F_zf, F_zr]
    Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]                                                                   # Fy = [F_yf, F_yr]

    eq1 = (Fy[1] * np.sin(x4) + u1 * np.cos(x4 - u0) + Fy[0] * np.sin(x4 - u0))/m                               # V dot
    eq2 = (Fy[1] * np.cos(x4) + Fy[0] * np.cos(x4 - u0) - u1 * np.sin(x4 - u0))/(m * x3) - x5                   # Beta dot
    eq3 = ((u1 * np.sin(u0) + Fy[0] * np.cos(u0)) * a - Fy[1] * b)/Iz                                           # Psi dot dot

    return [eq1, eq2, eq3]

# Initial guess for the solution
initial_guess = [0.5, 0.1, 300]          # [x5(0), u0(0), u1(0)]

# Use fsolve to find the solution

# FIRST EQUILIBRIUM
x3 = 7                  
x4 = 0 
eq[0,0] = np.copy(x3)
eq[1,0] = np.copy(x4)
eq[2:,0] = fsolve(equations, initial_guess)


# SECOND EQUILIBRIUM
x3 = 5                  
x4 = 0.25 
eq[0,1] = np.copy(x3)
eq[1,1] = np.copy(x4)
eq[2:,1] = fsolve(equations, initial_guess)

# Print the result
print('Equilibrium 1:', eq[0:,0], '\nEquilibrium 2:', eq[0:,1])


# Design REFERENCE TRAJECTORY  ---------------------------------------------------------------------------------------

TT = int(5.e1)
traj_ref = np.zeros((eq.shape[0], TT))

# Step reference signal - for all the states
T_mid = int((TT/2))

for tt in range(TT):
  if tt < T_mid:
    traj_ref[0, tt] = eq[0,0] 
    traj_ref[1, tt] = eq[1,0] 
    traj_ref[2, tt] = eq[2,0] 
    traj_ref[3, tt] = eq[3,0]
    traj_ref[4, tt] = eq[4,0]
  else:
    traj_ref[0, tt] = eq[0,1]
    traj_ref[1, tt] = eq[1,1] 
    traj_ref[2, tt] = eq[2,1]
    traj_ref[3, tt] = eq[3,1]
    traj_ref[4, tt] = eq[4,1]

tt_hor = range(TT)

fig, axs = plt.subplots(5, 1, sharex='all')

axs[0].plot(tt_hor, traj_ref[0,:], 'g--', linewidth=2)
#axs[0].plot(tt_hor, xx[0,:], linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_3$')

axs[1].plot(tt_hor, traj_ref[1,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_4$')

axs[2].plot(tt_hor, traj_ref[2,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$x_5$')

axs[3].plot(tt_hor, traj_ref[3,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$u_0$')

axs[4].plot(tt_hor, traj_ref[4,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$u_1$')
axs[4].set_xlabel('time')

fig.align_ylabels(axs)

plt.show()

# SMOOTHING the reference trajectory  -----------------------------------------------------------------------------------

# Perform linear interpolation for traj_ref[0]
new_num_points = 5  # Adjust the number of points for a smoother curve
interp_indices = np.linspace(0, TT - 1, new_num_points)
new_traj_ref_0 = np.interp(interp_indices, tt_hor, traj_ref[0, :])

# Plot the original traj_ref and the new trajectory for traj_ref[0]
plt.plot(tt_hor, traj_ref[0, :], 'g--', linewidth=2, label='Original traj_ref[0]')
plt.plot(interp_indices, new_traj_ref_0, '--', linewidth=2, label='Interpolated traj_ref[0]')
plt.grid()
plt.xlabel('time')
plt.ylabel('$x_3$')
plt.legend()
plt.show()

# GRADIENT METHOD evaluation  ----------------------------------------------------------------------------------------

# Define a cost function


Q = np.diag([1.0, 1.0, 1.0])
Q_f = np.array([[1, 0], [0, 1]])

r = 0.5
R = r*np.eye(ni)

QT = Q

def cost(xx,uu, xx_ref, uu_ref):
    """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
    - xx \in \R^2 state at time t
    - xx_ref \in \R^2 state reference at time t

    - uu \in \R^1 input at time t
    - uu_ref \in \R^2 input reference at time t


    Return 
    - cost at xx,uu
    - gradient of l wrt x, at xx,uu
    - gradient of l wrt u, at xx,uu

    """

    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]

    l = 0.5*(xx - xx_ref).T@Q@(xx - xx_ref) + 0.5*(uu - uu_ref).T@R@(uu - uu_ref)

    lx = Q@(xx - xx_ref)
    lu = R@(uu - uu_ref)

    return l.squeeze(), lx, lu

def cost_f(xx,xx_ref):
    """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
        - xx \in \R^2 state at time t
        - xx_ref \in \R^2 state reference at time t

    Return 
        - cost at xx,uu
        - gradient of l wrt x, at xx,uu
        - gradient of l wrt u, at xx,uu

    """
    xx = xx[:,None]
    xx_ref = xx_ref[:,None]

    lT = 0.5*(xx - xx_ref).T@QT@(xx - xx_ref)

    lTx = QT@(xx - xx_ref)

    return lT.squeeze(), lTx

# arrays to store data
xx = np.zeros((3, TT, max_iters))   # state seq.
uu = np.zeros((2, TT, max_iters))   # input seq.
xx_ref = np.zeros((3, TT))          # state ref.
uu_ref = np.zeros((2, TT))          # input ref.

lmbd = np.zeros((3, TT, max_iters))    # lambdas - costate seq.

deltau = np.zeros((2,TT, max_iters))   # Du - descent direction
dJ = np.zeros((2,TT, max_iters))       # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)                # collect cost
descent = np.zeros(max_iters)           # collect descent direction
descent_arm = np.zeros(max_iters)       # collect descent direction

# initial conditions
xx_init = np.zeros((3, TT))
uu_init = np.zeros((2, TT))

xx_ref = traj_ref[0:3]
uu_ref = traj_ref[3:]

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

print(np.shape(xx), np.shape(xx_ref))

for kk in range(max_iters-1):

    JJ[kk] = 0

    # calculate cost
    for tt in range(TT-1):
        temp_cost = cost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ[kk] += temp_cost

        temp_cost = cost_f(xx[:,-1,kk], xx_ref[:,-1])[0]
        JJ[kk] += temp_cost

    # Descent direction calculation
    lmbd_temp = cost_f(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

    for tt in reversed(range(TT-1)):                        # integration backward in time

        at, bt = cost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
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

    stepsize = 1

    for ii in range(armijo_maxiters):

        # temp solution update

        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = xx_ref[:,0]

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
            xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT-1):
            temp_cost = cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += temp_cost

        temp_cost = cost_f(xx_temp[:,-1], xx_ref[:,-1])[0]
        JJ_temp += temp_cost

        stepsizes.append(stepsize)                              # save the stepsize
        costs_armijo.append(np.min([JJ_temp, 100*JJ[kk]]))      # save the cost associated to the stepsize

        if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {:.3e}'.format(stepsize))
            break



print(JJ)