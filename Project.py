#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Dynamics import dynamics
from scipy.optimize import fsolve
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
max_iters = 30      #maximum number of iterations for Newton's method

m = 1480    #Kg
Iz = 1950   #Kg*m^2
a = 1.421   #m
b = 1.029   #m
mi = 1      #nodim
g = 9.81    #m/s^2

TT = int(5e2)          #discrete time samples
T_mid = TT/2            #half time
term_cond = 1e-6        #terminal condition

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
##### TASK 1: TRAJECTORY GENERATION I ###
#########################################

# We have to find the eqilibria for the system, a way to do that is to use the cornering equilibria, those associated to the systems with Betadot, Vdot and Psidotdot = 0
# Once I have set them I can focus on the last three equation, then imposing Veq and PsidotEq (I can think of this also as Veq/R with R a certain imposed radious) we obtain Betaeq, Fxeq and Deltaeq, in alternative I can set Veq and Betaeq and as concequence find the other eqilibrium values.
# The associated x and y trajectory can then be obtained by forward integration of the dynamics with the values we just found.
# For vehicles these trajectories are called corering eqilibria, in which I have circles with some radious and some Veq.

# Evaluate the EQUILIBRIUM  ----------------------------------------------------------------------------------

eq = np.zeros((ns+ni, 2))                                  

# calculation of the parameters at equilibrium
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

# FIRST EQUILIBRIUM
#imposing x3 and x4
x3 = 7                  
x4 = 0 

eq[2,0] = 0                                     # psi
eq[3,0] = np.copy(x3)                           # V
eq[4,0] = np.copy(x4)                           # beta
eq[5:,0] = fsolve(equations, initial_guess)     # psi dot, steering angle, force
eq[0,0]=(eq[3,0]*np.cos(eq[4,0])*np.cos(eq[2,0])-eq[3,0]*np.sin(eq[4,0])*np.sin(eq[2,0]))*T_mid     # x
eq[1,0]=(eq[3,0]*np.cos(eq[4,0])*np.sin(eq[2,0])+eq[3,0]*np.sin(eq[4,0])*np.cos(eq[2,0]))*T_mid     # y

# SECOND EQUILIBRIUM
x3 = 5                  
x4 = 0.1 

eq[2,1] = 20                                    
eq[3,1] = np.copy(x3)
eq[4,1] = np.copy(x4)
eq[5:,1] = fsolve(equations, initial_guess)
eq[0,1]= eq[0,0] + (eq[3,1]*np.cos(eq[4,1])*np.cos(eq[2,1])-eq[3,1]*np.sin(eq[4,1])*np.sin(eq[2,1]))*T_mid
eq[1,1]= eq[1,0] + (eq[3,0]*np.cos(eq[4,0])*np.sin(eq[2,1])+eq[3,0]*np.sin(eq[4,0])*np.cos(eq[2,1]))*T_mid

# Print the result
print('Equilibrium 1:', eq[0:,0], '\nEquilibrium 2:', eq[0:,1])


# Design REFERENCE TRAJECTORY  ---------------------------------------------------------------------------------------

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
    traj_ref[5, tt] = eq[5,0] 
    traj_ref[6, tt] = eq[6,0]
    traj_ref[7, tt] = eq[7,0]

  else:
    traj_ref[0, tt] = eq[0,1]
    traj_ref[1, tt] = eq[1,1] 
    traj_ref[2, tt] = eq[2,1]
    traj_ref[3, tt] = eq[3,1]
    traj_ref[4, tt] = eq[4,1]
    traj_ref[5, tt] = eq[5,1]
    traj_ref[6, tt] = eq[6,1]
    traj_ref[7, tt] = eq[7,1]

tt_hor = range(TT)

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

# GRADIENT METHOD evaluation  ----------------------------------------------------------------------------------------

# weight matrices
Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
QT = Q
R = np.diag([0.5, 0.5])

# arrays to store data
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.
xx_ref = np.zeros((ns, TT))          # state ref.
uu_ref = np.zeros((ni, TT))          # input ref.

# initial conditions
xx_init = np.zeros((ns, TT))
xx_init[3,:] = 1
uu_init = np.zeros((ni, TT))

xx_ref = traj_ref[0:6]
uu_ref = traj_ref[6:]

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

# perform Gradient Descent method
xx, uu, descent, JJ = Gradient (xx, uu, xx_ref, uu_ref, Q, R, QT, max_iters)

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


#########################################
##### TASK 2: TRAJECTORY GENERATION II ##
#########################################

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
