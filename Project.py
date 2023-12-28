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

xx = np.zeros((ns,))
ddx = np.zeros((ns,))
uu = np.zeros((ni,))
ddu = np.zeros((ni,))

for i in range (0,ns):
    ddx[i] = dx

for k in range (0,ni):
    ddu[k] = du

xx = x + ddx
xx_plus = dynamics(xx, u)[0]
diff_x = xx_plus - x_plus
check_x = diff_x - np.dot(A,ddx)

uu = u + ddu
xx_plus = dynamics(x, uu)[0]    
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


