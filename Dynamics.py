#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np

#define params
dt = 1e-3   #sample time
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