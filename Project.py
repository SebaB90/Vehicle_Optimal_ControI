#
# OPTCON PROJECT 
# Optimal Control of a Veichle
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 22/11/2022
#

import numpy as np

# TASK 0: DISCRETIZATION

ns = 6
ni = 2
dt = 1e-3

m = 1480 #Kg
Iz = 1950 #Kg*m^2
a = 1.421 #m
b = 1.029 #m
mi = 1 #nodim
g = 9.81 #m/s^2

Beta = [u[0] - (x[3]*np.sin(x[4]) + a*x[5])/(x[3]*np.sin(x[4])), - (x[3]*np.sin(x[4]) - b*x[5])/(x[3]*np.sin(x[4]))]    # Beta = [Beta_f, Beta_r]
Fz = [m*g*b/(a+b), m*g*a/(a+b)]     # Fz = [F_zf, F_zr]
Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]   # Fy = [F_yf, F_yr]

u = x = np.zeros((ni,))

x_plus = np.zeros((ns,))
x = np.zeros((ns,))

x_plus[0] = x[0] + dt*(x[3] * np.cos(x[4]) * np.cos(x[2]) - x[3] * np.sin(x[4]) * np.sin(x[2]))  
x_plus[1] = x[1] + dt*(x[3] * np.cos(x[4]) * np.sin(x[2]) + x[3] * np.sin(x[4]) * np.cos(x[2]))     # Y  dot
x_plus[2] = x[2] + dt*x[5]
x_plus[3] = x[3] + dt*((Fy[1] * np.sin(x[4]) + u[1] * np.cos(x[4] - u[0]) + Fy[0] * np.sin(x[4] - u[0]))/m)   # V dot
x_plus[4] = x[4] + dt*((Fy[1] * np.cos(x[4]) + Fy[0] * np.cos(x[4] - u[0]) - u[1] * np.sin(x[4] - u[0]))/(m * x[3]) - x[5])  # Beta dot
x_plus[5] = x[5] + dt*(((u[1] * np.sin(u[0]) + Fy[0] * np.cos(u[0])) * a - Fy[1] * b)/Iz)            # Psi dot dot


