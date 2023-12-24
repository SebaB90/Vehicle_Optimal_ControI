#
# Infinite-time LQR for regulation
# Lorenzo Sforni
# Bologna, 21/11/2022
#

import numpy as np
import matplotlib.pyplot as plt

import control as ctrl  #control package python


# Import mass-spring-damper cart dynamics
import OPCON_LAB64_Dynamics as dyn

# Import LTI LQR solver
from OPCON_LAB65_solver_lti_LQR import lti_LQR

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})



#######################################
# Parameters
#######################################

tf = 1 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics

TT = int(tf/dt) # discrete-time samples


#######################################
# Dynamics
#######################################

ns = 2
ni = 1

x0 = np.array([2, 1])

xdummy = np.array([0, 0])
udummy = np.array([0])

fx,fu = dyn.dynamics(xdummy,udummy)[1:]

AA = fx.T
BB = fu.T


#######################################
# Cost
#######################################

QQ = np.array([[1e2, 0], [0, 1]])
RR = 1*np.eye(ni)

# different possibilities

QQf = np.array([[1e2, 0], [0, 100]])


# Infinite-horizon gain
GG = ctrl.dare(AA,BB,QQ,RR)[-1]

KK_inf = -GG


#######################################
# Main
#######################################

xx_inf = np.zeros((ns,TT))
uu_inf = np.zeros((ni,TT))

KK_fin = lti_LQR(AA,BB,QQ,RR,QQf,TT)[0] # for comparison
xx_fin = np.zeros((ns,TT))
uu_fin = np.zeros((ni,TT))

xx_inf[:,0] = x0
xx_fin[:,0] = x0

for tt in range(TT-1):
  
  # infinite hor
  uu_inf[:,tt] = KK_inf@xx_inf[:,tt]
  xx_inf[:,tt+1] = AA@xx_inf[:,tt] + BB@uu_inf[:,tt]

  # finite hor
  uu_fin[:,tt] = KK_fin[:,:,tt]@xx_fin[:,tt]
  xx_fin[:,tt+1] = AA@xx_fin[:,tt] + BB@uu_fin[:,tt]


#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_fin[0,:],'b--' ,linewidth=2, label = 'fin')
axs[0].plot(tt_hor, xx_inf[0,:],'b',linewidth=2, label = 'inf')
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_fin[1,:],'b--', linewidth=2, label = 'fin')
axs[1].plot(tt_hor, xx_inf[1,:],'b',linewidth=2, label = 'inf')

axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu_fin[0,:],'r--', linewidth=2, label = 'fin')
axs[2].plot(tt_hor, uu_inf[0,:],'r', linewidth=2, label = 'inf')
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')


fig.align_ylabels(axs)







plt.show()







