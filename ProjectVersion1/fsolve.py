
import numpy as np

x = [6.50409711, 0.90841421]

v1 = x[0] * np.cos(x[1]) - 4
v2 = x[1] * x[0] - x[1] - 5

# print (v1, v2)

# Vehicle parameters
mm = 1480  # Kg
Iz = 1950  # Kg*mm^2
aa = 1.421  # mm           
bb = 1.029  # mm
mi = 1  # nodim
gg = 9.81  # mm/s^2
x3 = 3                  
x4 = 0
vars = [0, 0, 4.44000000e+03]

Beta = [vars[1] - (x3*np.sin(x4) + aa*vars[0])/(x3*np.cos(x4)), - (x3*np.sin(x4) - bb*vars[0])/(x3*np.cos(x4))]               # Beta = [Beta_f, Beta_r]
Fz = [mm*gg*bb/(aa+bb), mm*gg*aa/(aa+bb)]                                                                         # Fz = [F_zf, F_zr]
Fy = [mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]]

v3 = ((Fy[1] * np.sin(x4) + vars[2] * np.cos(x4 - vars[1]) + Fy[0] * np.sin(x4 - vars[1]))/mm) - x3
v4 = (Fy[1] * np.cos(x4) + Fy[0] * np.cos(x4 - vars[1]) - vars[2] * np.sin(x4 - vars[1]))/(mm * x3) - vars[0] - x4
v5 = ((vars[2] * np.sin(vars[1]) + Fy[0] * np.cos(vars[1])) * aa - Fy[1] * bb)/Iz - vars[0]

print (v3, v4, v5)

a = (vars[2]/mm)- x3

print ('\n',a)