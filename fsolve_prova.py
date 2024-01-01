import numpy as np
from scipy.optimize import fsolve


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
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.
xx_ref = np.zeros((ns, TT))          # state ref.
uu_ref = np.zeros((ni, TT))          # input ref.

Qt = np.diag([1, 1, 100, 1, 100, 100])
Q = Qt
R = np.diag([1, 0.01])
xx = xx[:,None]
uu = uu[:,None]

xx_ref = xx_ref[:,None]
uu_ref = uu_ref[:,None]

l = 0.5*(xx - xx_ref).T@Q@(xx - xx_ref) + 0.5*(uu - uu_ref).T@R@(uu - uu_ref)

lx = Q@(xx - xx_ref)
lu = R@(uu - uu_ref)

lxx = Q
luu = R
lux = np.zeros((ns, ni))
lxu = np.zeros((ni, ns))

hessian = np.block([[lxx, lux], [lxu, luu]])
gradient = np.concatenate([lx, lu], axis=0)

print(np.shape(gradient), np.shape(lx))