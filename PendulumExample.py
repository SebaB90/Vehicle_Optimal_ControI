#
# Pendulum Example
# 
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 27/12/2023
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

dt = 1e-3   #sample time

ns = 2      #number of states
ni = 1      #number of inputs

m = 1       #Kg
g = 9.81    #m/s^2
l = 1       #m

xx = np.zeros((ns,1))
uu = np.zeros((ni,1))


########################
# Linear Dynamics - get nominal A,B matrices
########################

def dynamics (x, u):
    x_plus = np.zeros((ns,))
    x_plus[0] = x[0] + dt*(x[1])                                 # x1 dot
    x_plus[1] = x[1] + dt*(-g/l*np.sin(x[0]) + 1/(m*l**2)*u)     # x2 dot
    fx = np.array([[1, dt*(-g/l*np.cos(x[0]))],
                   [dt, 1]])
    fu = np.array([[0, dt*(1/(m*l**2))]])
    return x_plus, fx, fu

fx, fu = dynamics(xx,uu)[1:]
AAnom = fx.T  # nominal A
BBnom = fu.T  # nominal B

# OPEN LOOP TEST to check if the dynamics do what expected ---------------------------------------------------
x1_traj = [xx[0]]
x2_traj = [xx[1]]

total_time = 10                     # Adjust the total simulation time as needed
num_steps = int(total_time / dt)

for _ in range(num_steps):
    x, _, _ = dynamics(xx, uu)
    x1_traj.append(xx[0])
    x2_traj.append(xx[1])

# Plotting the trajectory
plt.plot(x1_traj, x2_traj, label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()



########################
# Compute the equilibtium of the system
########################

##############################
# Reference trajectory
##############################

TT = int(5.e1)
xx_ref = np.zeros((ns, TT))

# Step reference signal - for all the states
T_mid = int((TT/2))

for tt in range(TT):
  if tt < T_mid:
    xx_ref[0, tt] = x3eq1, xx_ref[1, tt] = x4eq1, xx_ref[2, tt] = x5eq1, xx_ref[3, tt] = u1eq1, xx_ref[4, tt] = u2eq1
  else:
    xx_ref[0, tt] = x3eq2, xx_ref[1, tt] = x4eq2, xx_ref[2, tt] = x5eq2, xx_ref[3, tt] = u1eq2, xx_ref[4, tt] = u2eq2

    
tt_hor = range(TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
#axs[0].plot(tt_hor, xx[0,:], linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_3$')

axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_4$')

axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_5$')

axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$u_0$')

axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
#axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$u_1$')
axs[2].set_xlabel('time')

fig.align_ylabels(axs)

plt.show()
