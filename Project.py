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
from Dynamics import dynamics
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
from Newton import Newton
from Gradient import Gradient


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

##################################
##### TASK 0: DISCRETIZATION #####
##################################


#define params

dt = 1e-3           #sample time
dx = 1e-3           #infinitesimal increment
du = 1e-3           #infinitesimal increment
ns = 6              #number of states
ni = 2              #number of inputs
max_iters = 20      #maximum number of iterations for Newton's method

m = 1480    #Kg
Iz = 1950   #Kg*m^2
a = 1.421   #m
b = 1.029   #m
mi = 1      #nodim
g = 9.81    #m/s^2

TT = int(5e2)          #discrete time samples
T_mid = TT/2            #half time
term_cond = 1e-6        #terminal condition

plot = True
Task0 = False

if Task0 :
    # defining x and u
    u = np.array([0.25, 20])
    x = np.array([0, 0, 0, 1, 0, 0])

    x_plus, fx, fu = dynamics(x, u)

    A = fx.T
    B = fu.T

    # OPEN LOOP TEST to check if the dynamics do what expected ---------------------------------------------------
    if plot:
        x_traj = [np.copy(x[0])]
        y_traj = [np.copy(x[1])]
        traj = np.copy(x)

        total_time = 100                     # Adjust the total simulation time as needed
        num_steps = int(total_time / dt)

        for _ in range(num_steps):
            traj = dynamics(traj, u)[0]
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
##### TASK 1: TRAJECTORY GENERATION I ###
#########################################

# We have to find the eqilibria for the system, a way to do that is to use the cornering equilibria, those associated to the systems with Betadot, Vdot and Psidotdot = 0
# Once I have set them I can focus on the last three equation, then imposing Veq and PsidotEq (I can think of this also as Veq/R with R a certain imposed radious) we obtain Betaeq, Fxeq and Deltaeq, in alternative I can set Veq and Betaeq and as concequence find the other eqilibrium values.
# The associated x and y trajectory can then be obtained by forward integration of the dynamics with the values we just found.
# For vehicles these trajectories are called corering eqilibria, in which I have circles with some radious and some Veq.

# Evaluate the EQUILIBRIUM  ----------------------------------------------------------------------------------

eq = np.zeros((ns+ni, 2))
initial_guess = [0.5, 0.1, 0]          # [x5(0), u0(0), u1(0)]

# calculation of the parameters at equilibrium
def equations(vars):
    x5, u0, u1 = vars
    Beta = [u0 - (x3*np.sin(x4) + a*x5)/(x3*np.cos(x4)), - (x3*np.sin(x4) - b*x5)/(x3*np.cos(x4))]              # Beta = [Beta_f, Beta_r]
    Fz = [m*g*b/(a+b), m*g*a/(a+b)]                                                                             # Fz = [F_zf, F_zr]
    Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]                                                                   # Fy = [F_yf, F_yr]

    eq1 = (Fy[1] * np.sin(x4) + u1 * np.cos(x4 - u0) + Fy[0] * np.sin(x4 - u0))/m                               # V dot (x3)
    eq2 = (Fy[1] * np.cos(x4) + Fy[0] * np.cos(x4 - u0) - u1 * np.sin(x4 - u0))/(m * x3) - x5                   # Beta dot (x4)
    eq3 = ((u1 * np.sin(u0) + Fy[0] * np.cos(u0)) * a - Fy[1] * b)/Iz                                           # Psi dot dot (x5)

    return [eq1, eq2, eq3]

# Initial guess for the solution

# FIRST EQUILIBRIUM
#imposing x3 and x4
x3 = 1                  
x4 = 0

eq[3,0] = np.copy(x3)                           # V
eq[4,0] = np.copy(x4)                           # beta
eq[5:,0] = fsolve(equations, initial_guess)     # psi dot, steering angle, force
eq[2,0] = eq[5,0]                               # psi   
eq[0,0] =(eq[3,0]*np.cos(eq[4,0])*np.cos(eq[2,0])-eq[3,0]*np.sin(eq[4,0])*np.sin(eq[2,0]))     # x
eq[1,0] =(eq[3,0]*np.cos(eq[4,0])*np.sin(eq[2,0])+eq[3,0]*np.sin(eq[4,0])*np.cos(eq[2,0]))     # y

# SECOND EQUILIBRIUM
x3 = 5                 
x4 = 0.25

eq[3,1] = np.copy(x3)
eq[4,1] = np.copy(x4)
eq[5:,1] = fsolve(equations, initial_guess)
eq[2,1] = eq[5,1]*T_mid    
eq[0,1] = (eq[3,1]*np.cos(eq[4,1])*np.cos(eq[2,1])-eq[3,1]*np.sin(eq[4,1])*np.sin(eq[2,1]))
eq[1,1] = (eq[3,1]*np.cos(eq[4,1])*np.sin(eq[2,1])+eq[3,1]*np.sin(eq[4,1])*np.cos(eq[2,1]))


# Print the result
print('Equilibrium 1:', eq[0:,0], '\nEquilibrium 2:', eq[0:,1])


# Design REFERENCE TRAJECTORY  ---------------------------------------------------------------------------------------

traj_ref = np.zeros((ns+ni, TT))
traj_ref[:,0] = eq[:,0]

# Step reference signal - for all the states
T_mid = int((TT/2))

for tt in range(1,TT):
  
  traj = dynamics(traj_ref[:,tt-1], traj_ref[6:,tt-1])[0]

  if tt <= TT/2:
    traj_ref[0, tt] = traj[0]
    traj_ref[1, tt] = traj[1]
    traj_ref[2, tt] = traj[2]
    traj_ref[3:, tt] = eq[3:,0]

  else:
    traj_ref[0, tt] = traj[0]  
    traj_ref[1, tt] = traj[1]
    traj_ref[2, tt] = traj[2]
    traj_ref[3:, tt] = eq[3:,1]

tt_hor = range(TT)

####################################################################################################################################
plt.plot(traj_ref[0,:], traj_ref[1,:], label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
#######################################################################################################################################

fig, axs = plt.subplots(8, 1, sharex='all')

axs[0].plot(tt_hor, traj_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x$')

axs[1].plot(tt_hor, traj_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$y$')

axs[2].plot(tt_hor, traj_ref[2,:], 'g--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$psi$')

axs[3].plot(tt_hor, traj_ref[3,:], 'g--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$V$')

axs[4].plot(tt_hor, traj_ref[4,:], 'g--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$beta$')

axs[5].plot(tt_hor, traj_ref[5,:], 'g--', linewidth=2)
axs[5].grid()
axs[5].set_ylabel('$psi dot$')

axs[6].plot(tt_hor, traj_ref[6,:], 'g--', linewidth=2)
axs[6].grid()
axs[6].set_ylabel('$u_0$')

axs[7].plot(tt_hor, traj_ref[7,:], 'g--', linewidth=2)
axs[7].grid()
axs[7].set_ylabel('$u_1$')
axs[7].set_xlabel('time')

fig.align_ylabels(axs)

plt.show()

# NEWTON'S METHOD evaluation  ----------------------------------------------------------------------------------------

# weight matrices
Qt = 0.1*np.diag([1, 1, 10, 1, 10, 10])
QT = Qt
Rt = np.diag([10, 1])

# arrays to store data
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.
xx_ref = np.zeros((ns, TT))          # state ref.
uu_ref = np.zeros((ni, TT))          # input ref.

# initial conditions
xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

for i in range(0,TT):
    xx_init[:,i] = traj_ref[0:6,0]
    uu_init[:,i] = traj_ref[6:,0]

xx_ref = traj_ref[0:6,:]
uu_ref = traj_ref[6:,:]

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

# perform Newton's like method
if plot:
    xx, uu, descent, JJ = Gradient(xx, uu, xx_ref, uu_ref, Qt, Rt, QT, max_iters)

    xx_star = xx[:,:,max_iters-1]
    uu_star = uu[:,:,max_iters-1]
    uu_star[:,-1] = uu_star[:,-2]        # for plotting purposes

    # Plots

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters), descent[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    plt.figure('cost')
    plt.plot(np.arange(max_iters), JJ[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

# Design OPTIMAL TRAJECTORY  ---------------------------------------------------------------------------------------
if plot:
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
    axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x$')

    axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
    axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y$')

    axs[2].plot(tt_hor, xx_star[2,:],'r', linewidth=2)
    axs[2].plot(tt_hor, xx_ref[2,:], 'r--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$psi$')

    axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
    axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$V$')

    axs[4].plot(tt_hor, xx_star[4,:], linewidth=2)
    axs[4].plot(tt_hor, xx_ref[4,:], 'g--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$beta$')

    axs[5].plot(tt_hor, xx_star[5,:],'r', linewidth=2)
    axs[5].plot(tt_hor, xx_ref[5,:], 'r--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$psi dot$')

    axs[6].plot(tt_hor, uu_star[0,:], linewidth=2)
    axs[6].plot(tt_hor, uu_ref[0,:], 'g--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$delta$')

    axs[7].plot(tt_hor, uu_star[1,:],'r', linewidth=2)
    axs[7].plot(tt_hor, uu_ref[1,:], 'r--', linewidth=2)
    axs[7].grid()
    axs[7].set_ylabel('$F$')
    axs[7].set_xlabel('time')

    plt.show()

    # Plotting the trajectory
    plt.plot(xx_star[0,:], xx_star[1,:], label='Trajectory')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

#########################################
##### TASK 2: TRAJECTORY GENERATION II ##
#########################################

# SMOOTHING the reference trajectory  -----------------------------------------------------------------------------------

# Perform linear interpolation for reference trajectory
fig, axs = plt.subplots(8, 1, sharex='all')
fig.suptitle('Trajectory Smoothing using PCHIP Spline')
traj_smooth = np.zeros((8,TT))
x_traj_smooth = np.zeros((8,TT))

for i in range (ns+ni):
    new_num_points = 7      # Adjust the number of points for a smoother curve
    interp_indices = np.linspace(0, TT - 1, new_num_points)
    new_traj_ref_0 = np.interp(interp_indices, tt_hor, traj_ref[i,:])

    # define point to create spline
    x_spl = np.array([interp_indices[0].astype(int), interp_indices[1].astype(int), interp_indices[2].astype(int), interp_indices[4].astype(int), interp_indices[5].astype(int), interp_indices[6].astype(int)])
    y_spl = np.array([new_traj_ref_0[0], new_traj_ref_0[0], new_traj_ref_0[0], new_traj_ref_0[-1], new_traj_ref_0[-1], new_traj_ref_0[-1]])

    # Create a piecewise cubic Hermite interpolating polynomial(PCHIP) interpolation of the given points
    cs = PchipInterpolator(x_spl, y_spl)

    # Generate new, smoother x values (denser for plotting)
    x_spl_new = np.linspace(min(x_spl), max(x_spl), TT)

    # Compute the smoothed y values
    y_spl_new = cs(x_spl_new)

    # Store the values inside an array
    traj_smooth[i,:] = y_spl_new

    # Plotting the original and smoothed trajectories
    axs[i].plot(tt_hor, traj_ref[i, :], 'g--', linewidth=2, label='Original Reference Trajectory')
    axs[i].plot(interp_indices, new_traj_ref_0, 'b--', linewidth=2, label='Interpolated Trajectory')
    axs[i].plot(x_spl, y_spl, 'o', label='Points used for spline creation')
    axs[i].plot(x_spl_new, y_spl_new, 'r-', label='Smoothed Trajectory')
    axs[i].grid()
    if i == 8:
        axs[i].xlabel('time')

axs[0].set_ylabel('$x$')
axs[1].set_ylabel('$y$')
axs[2].set_ylabel('$psi$')
axs[3].set_ylabel('$V$')
axs[4].set_ylabel('$beta$')
axs[5].set_ylabel('$psi dot$')
axs[6].set_ylabel('$delta$')
axs[7].set_ylabel('$F$')
plt.legend()
plt.show()

# initial conditions
xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

for i in range(0,TT):
    xx_init[:,i] = traj_smooth[0:6,0]
    uu_init[:,i] = traj_smooth[6:,0]

xx_ref = traj_smooth[0:6,:]
uu_ref = traj_smooth[6:,:]

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

xx, uu, descent, JJ = Gradient(xx, uu, xx_ref, uu_ref, Qt, Rt, QT, max_iters)

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2]        # for plotting purposes

# Plots

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# Design OPTIMAL TRAJECTORY  ---------------------------------------------------------------------------------------

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$y$')

axs[2].plot(tt_hor, xx_star[2,:],'r', linewidth=2)
axs[2].plot(tt_hor, xx_ref[2,:], 'r--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$psi$')

axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$V$')

axs[4].plot(tt_hor, xx_star[4,:], linewidth=2)
axs[4].plot(tt_hor, xx_ref[4,:], 'g--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$beta$')

axs[5].plot(tt_hor, xx_star[5,:],'r', linewidth=2)
axs[5].plot(tt_hor, xx_ref[5,:], 'r--', linewidth=2)
axs[5].grid()
axs[5].set_ylabel('$psi dot$')

axs[6].plot(tt_hor, uu_star[0,:], linewidth=2)
axs[6].plot(tt_hor, uu_ref[0,:], 'g--', linewidth=2)
axs[6].grid()
axs[6].set_ylabel('$delta$')

axs[7].plot(tt_hor, uu_star[1,:],'r', linewidth=2)
axs[7].plot(tt_hor, uu_ref[1,:], 'r--', linewidth=2)
axs[7].grid()
axs[7].set_ylabel('$F$')
axs[7].set_xlabel('time')

plt.show()

# Plotting the trajectory
plt.plot(xx_star[0,:], xx_star[1,:], label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()