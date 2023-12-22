#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np
import matplotlib.pyplot as plt

##### TASK 0: DISCRETIZATION #####


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

    x_plus = np.ones((ns,))
    Beta = [u[0] - (x[3]*np.sin(x[4]) + a*x[5])/(x[3]*np.cos(x[4])), - (x[3]*np.sin(x[4]) - b*x[5])/(x[3]*np.cos(x[4]))]    # Beta = [Beta_f, Beta_r]
    Fz = [m*g*b/(a+b), m*g*a/(a+b)]             # Fz = [F_zf, F_zr]
    Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]   # Fy = [F_yf, F_yr]

    x_plus[0] = x[0] + dt*(x[3] * np.cos(x[4]) * np.cos(x[2]) - x[3] * np.sin(x[4]) * np.sin(x[2]))                 # X dot
    x_plus[1] = x[1] + dt*(x[3] * np.cos(x[4]) * np.sin(x[2]) + x[3] * np.sin(x[4]) * np.cos(x[2]))                 # Y dot
    x_plus[2] = x[2] + dt*x[5]                                                                                      # Psi dot
    x_plus[3] = x[3] + dt*((Fy[1] * np.sin(x[4]) + u[1] * np.cos(x[4] - u[0]) + Fy[0] * np.sin(x[4] - u[0]))/m)     # V dot
    x_plus[4] = x[4] + dt*((Fy[1] * np.cos(x[4]) + Fy[0] * np.cos(x[4] - u[0]) - u[1] * np.sin(x[4] - u[0]))/(m * x[3]) - x[5])  # Beta dot
    x_plus[5] = x[5] + dt*(((u[1] * np.sin(u[0]) + Fy[0] * np.cos(u[0])) * a - Fy[1] * b)/Iz)                       # Psi dot dot

    fx = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [dt*(-x[3] * np.cos(x[4]) * np.sin(x[2]) - x[3] * np.sin(x[4]) * np.cos(x[2])), dt*(x[3] * np.cos(x[4]) * np.cos(x[2]) - x[3] * np.sin(x[4]) * np.sin(x[2])), 1, 0, 0, 0],
                  [dt*(np.cos(x[4]) * np.cos(x[2]) - np.sin(x[4]) * np.sin(x[2])), dt*(np.cos(x[4]) * np.sin(x[2]) + np.sin(x[4]) * np.cos(x[2])), 0, 1, -(1/(m*(x[3]**2)))*dt*((Fy[1] * np.cos(x[4]) + Fy[0] * np.cos(x[4] - u[0]) - u[1] * np.sin(x[4] - u[0]))), 0],
                  [dt*(x[3] * -np.sin(x[4]) * np.cos(x[2]) - x[3] * np.cos(x[4]) * np.sin(x[2])), dt*(x[3] * -np.sin(x[4]) * np.sin(x[2]) + x[3] * np.cos(x[4]) * np.cos(x[2])), 0, dt*((Fy[1] * np.cos(x[4]) + u[1] * - np.sin(x[4] - u[0]) + Fy[0] * np.cos(x[4] - u[0]))/m), 1 + dt*((Fy[1] * -np.sin(x[4]) + Fy[0] * np.sin(x[4] - u[0]) + u[1] * np.cos(x[4] - u[0]))/(m * x[3])), 0],
                  [0, 0, dt, 0, -1, 1]])
    
    fu = np.array([[0, 0, 0, dt*(u[1] * np.sin(x[4] - u[0]) + Fy[0] * -np.cos(x[4] - u[0]))/m, dt*(Fy[0] * np.sin(x[4] - u[0]) + u[1] * np.cos(x[4] - u[0]))/(m * x[3]),  dt*((u[1] * np.cos(u[0]) - Fy[0] * np.sin(u[0])) *a/Iz)],
                  [0, 0, 0, dt*np.cos(x[4] - u[0])/m, dt*(- np.sin(x[4] - u[0]))/(m * x[3]), dt * np.sin(u[0]) * a/Iz]])

    return x_plus, fx, fu


##### TASK 1: TRAJECTORY GENERATION #####

# We have to find the eqilibria for the system, a way to do that is to use the cornering equilibria, those associated to the systems with Betadot, Vdot and Psidodot = 0
# Once I have set them I can concentrate on the last three equation, then imposing Veq and PsidotEq (I can think of this also as Veq/R with R a certain imposed radious) we obtain Betaeq, Fxeq and Deltaeq, in alternative I can set Veq and Betaeq and as concequence find the other eqilibrium values.
# The associated x and y trajectory can then be obtained by forward integration of the dynamics with the values we just found.
# For vehicles these trajectories are called corering eqilibria, in which I have circles with some radious and some Veq.

u = np.array([0, 0])
x = np.array([10, 0, 0, 1, 0, 0])

x_plus, fx, fu = dynamics(x, u)

A = fx.T
B = fu.T

x_traj = [x[0]]
y_traj = [x[1]]

total_time = 10  # Adjust the total simulation time as needed
num_steps = int(total_time / dt)

for _ in range(num_steps):
    x, _, _ = dynamics(x, u)
    x_traj.append(x[0])
    y_traj.append(x[1])

# Plotting the trajectory
plt.plot(x_traj, y_traj, label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()


xx = np.zeros((ns,))
ddx = np.zeros((ns,))

for i in range (0,ns):
    ddx[i] = dx

xx = x + ddx
xx_plus = dynamics(xx, u)[0]
diff = xx_plus - x_plus
check = diff - np.dot(fx,ddx)

print (check)
