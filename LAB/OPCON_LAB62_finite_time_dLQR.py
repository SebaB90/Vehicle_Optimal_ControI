#
# Finite-time LQR for regulation
# Lorenzo Sforni
# Bologna, 21/11/2022
#

import numpy as np
import matplotlib.pyplot as plt

import control as ctrl  #control package python


# Import mass-spring-damper cart dynamics
import OPCON_LAB64_Dynamics as dyn

# Import LTI LQR solver
from OPCON_LAB54_SOLVER_LTI_LQR import lti_LQR

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
RR = 1e-2*np.eye(ni)

# different possibilities
print(AA.shape,QQ.shape)
print(BB.shape,RR.shape)

QQf = np.eye(ns)
#QQf = ctrl.dare(AA,BB,QQ,RR)[0]


#######################################
# Main
#######################################


KK,PP = lti_LQR(AA,BB,QQ,RR,QQf,TT)

xx = np.zeros((ns,TT))
uu = np.zeros((ni,TT))

xx[:,0] = x0

for tt in range(TT-1):
  uu[:,tt] = KK[:,:,tt]@xx[:,tt]
  xx[:,tt+1] = AA@xx[:,tt] + BB@uu[:,tt]


#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx[0,:], linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu[0,:],'r', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')

fig.align_ylabels(axs)

plt.figure()

for ii in range(ns):
  for jj in range(ns):
    plt.plot(tt_hor, PP[ii,jj,:])

plt.grid()
plt.ylabel('$P_{i,j}$')
plt.xlabel('time')





plt.show()







