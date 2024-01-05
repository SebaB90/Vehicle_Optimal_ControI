#
# Finite-time LQR for tracking
# Lorenzo Sforni
# Bologna, 20/11/2023
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})


from OPCON_LAB72_solver_ltv_LQR import ltv_LQR

ns = 2
ni = 1

TT = int(5.e1)

# System dynamics - double integrator

AA = np.array([[1,1],[0, 1]])
BB = np.array([[0], [1]])

##############################
# Reference trajectory
##############################

xx_ref = np.zeros((ns, TT))

# Step reference signal - for first state only
T_mid = int((TT/2))

for tt in range(TT):
  if tt < T_mid:
    xx_ref[0, tt] = 0
  else:
    xx_ref[0, tt] = 10


##############################
# Cost 
##############################

QQ = np.array([[10, 0], [0, 1]])
QQ_f = np.array([[10, 0], [0, 1]])

r = 0.5
RR = r*np.eye(ni)

SS = np.zeros((ni,ns))

# Affine terms (for tracking)

qq = np.zeros((ns,TT))
rr = np.zeros((ni,TT))

for tt in range(TT):
    qq_temp = -QQ@xx_ref[:,tt]
    qq[:,tt] = qq_temp.squeeze()

qqf =  -QQ_f@xx_ref[:,-1]

##############################
# Solver 
##############################

# initial condition
x0 = np.array([0, 0])

KK,sigma = ltv_LQR(AA,BB,QQ,RR,SS,QQ_f, TT, x0, qq, rr, qqf)[:2]

xxx,uuu = ltv_LQR(AA,BB,QQ,RR,SS,QQ_f, TT, x0, qq, rr, qqf)[3:]
print(np.shape(xxx),np.shape(uuu))

xx = np.zeros((ns, TT))
uu = np.zeros((ni, TT))

for tt in range(TT-1): 
  #
  # Trajectory
  #
  uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
  xx_p = AA@xx[:,tt] + BB@uu[:, tt]
  #
  xx[:,tt+1] = xx_p
  #


#######################################
# Plots
#######################################

tt_hor = range(TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].plot(tt_hor, xx[0,:], linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].plot(tt_hor, xx[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu[0,:],'r', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')

fig.align_ylabels(axs)

plt.show()