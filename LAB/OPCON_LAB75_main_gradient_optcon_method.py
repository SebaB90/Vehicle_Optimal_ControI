#
# Gradient method for Optimal Control
# Main
# Lorenzo Sforni
# Bologna, 22/11/2022
#


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import pendulum dynamics
import OPCON_LAB74_dynamics as dyn

# import cost functions
import OPCON_LAB73_cost as cst

from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

#######################################
# Algorithm parameters
#######################################

max_iters = int(10)
stepsize_0 = 1

# ARMIJO PARAMETERS
cc = 0.5
# beta = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

term_cond = 1e-6

visu_armijo = False
visu_animation = True

#######################################
# Trajectory parameters
#######################################

tf = 10 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt) # discrete-time samples


######################################
# Reference curve
######################################

ref_deg_0 = 0
ref_deg_T = 30
#
# 

step_reference = False

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

if not step_reference:
  #
  # Generate a bell-shaped reference curve
  #

  # Curve parameters
  #
  a = 1
  b = tf/2 + dt
  #
  # initial and final conditions
  #
  x0 = 0*np.ones((1,))
  xf = 30*np.pi/180
  #
  # sampling time
  #
  t_samp_temp = np.linspace(dt, 2*b - dt, int(tf/dt))
  #
  #
  def bell_function(t, x, a):
      tau = t - b
      exp_arg = -(2*b**2)*(tau**2 + b**2)/(tau**2 - b**2)**2
      beta = a * np.exp(exp_arg)
      return beta*np.ones((1,))

  # 
  sigma_temp = solve_ivp(bell_function, (dt, 2*b - dt), np.zeros((1,)), t_eval = t_samp_temp, args = (1,), method = 'Radau')
  norm_factor = sigma_temp.y[0][-1] # normalization factor
  #
  sigma = solve_ivp(bell_function, (dt, 2*b - dt), x0, t_eval = t_samp_temp, args = ((xf-x0)/norm_factor,), method = 'Radau').y[0]

  t_samp = t_samp_temp - dt

  KKeq = dyn.KKeq

  xx_ref[0] = sigma
  uu_ref[0,:] = KKeq*np.sin(xx_ref[0,:])

else:

  KKeq = dyn.KKeq

  xx_ref[0,int(TT/2):] = np.ones((1,int(TT/2)))*np.ones((1,int(TT/2)))*np.deg2rad(ref_deg_T)
  uu_ref[0,:] = KKeq*np.sin(xx_ref[0,:])
   

x0 = xx_ref[:,0]

######################################
# Initial guess
######################################

xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))     # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

print(np.shape(xx), np.shape(xx_ref))

for kk in range(max_iters-1):

  JJ[kk] = 0
  # calculate cost
  for tt in range(TT-1):
    temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
    JJ[kk] += temp_cost
  
  temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
  JJ[kk] += temp_cost


  ##################################
  # Descent direction calculation
  ##################################

  lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

  for tt in reversed(range(TT-1)):  # integration backward in time

    at, bt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
    fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]

    At = fx.T
    Bt = fu.T


    lmbd_temp = At.T@lmbd[:,tt+1,kk][:,None] + at       # costate equation
    dJ_temp = Bt.T@lmbd[:,tt+1,kk][:,None] + bt         # gradient of J wrt u
    deltau_temp = - dJ_temp


    lmbd[:,tt,kk] = lmbd_temp.squeeze()
    dJ[:,tt,kk] = dJ_temp.squeeze()
    deltau[:,tt,kk] = deltau_temp.squeeze()

    descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]
    descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk]


  ##################################
  # Stepsize selection - ARMIJO
  ##################################


  stepsizes = []  # list of stepsizes
  costs_armijo = []

  stepsize = stepsize_0

  for ii in range(armijo_maxiters):

    # temp solution update

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

    # temp cost calculation
    JJ_temp = 0

    for tt in range(TT-1):
      temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ_temp += temp_cost

    temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost

    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(np.min([JJ_temp, 100*JJ[kk]]))    # save the cost associated to the stepsize

    if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
        # update the stepsize
        stepsize = beta*stepsize
    
    else:
        print('Armijo stepsize = {:.3e}'.format(stepsize))
        break

  # plt.plot(xx_temp[0,:])
  # plt.show()
  ############################
  # Armijo plot
  ############################

  if visu_armijo and kk%10 == 0:

    steps = np.linspace(0,stepsize_0,int(2e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

      step = steps[ii]

      # temp solution update

      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
      JJ_temp += temp_cost

      costs[ii] = np.min([JJ_temp, 100*JJ[kk]])


    plt.figure(1)
    plt.clf()

    plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    # plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()

    plt.show()

#     plt.pause(4)

    
  ############################
  # Update the current solution
  ############################


  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))

  xx_temp[:,0] = x0

  for tt in range(TT-1):
    uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

  xx[:,:,kk+1] = xx_temp
  uu[:,:,kk+1] = uu_temp

  ############################
  # Termination condition
  ############################

  print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk

    break

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

############################
# Plots
############################

# cost and descent

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


# optimal trajectory


tt_hor = np.linspace(0,tf,TT)


fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
axs[2].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')
  

plt.show()

#####################################
# Animation
#####################################

import matplotlib.animation as animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) #minor grid

dt = 0.001

time = np.arange(len(tt_hor))*dt

if visu_animation:
    
  fig = plt.figure()
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-.5, 1.2))
  ax.grid()
  # no labels
  ax.set_yticklabels([])
  ax.set_xticklabels([])

  
  line0, = ax.plot([], [], 'o-', lw=2, c='b', label='Optimal')
  line1, = ax.plot([], [], '*-', lw=2, c='g',dashes=[2, 2], label='Reference')

  time_template = 't = %.1f s'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
  fig.gca().set_aspect('equal', adjustable='box')

  # Subplot
  left, bottom, width, height = [0.64, 0.13, 0.2, 0.2]
  ax2 = fig.add_axes([left, bottom, width, height])
  ax2.xaxis.set_major_locator(MultipleLocator(2))
  ax2.yaxis.set_major_locator(MultipleLocator(0.25))
  ax2.set_xticklabels([])
  

  ax2.grid(which='both')
  ax2.plot(time, xx_star[0],c='b')
  ax2.plot(time, xx_ref[0], color='g', dashes=[2, 1])

  point1, = ax2.plot([], [], 'o', lw=2, c='b')


  def init():
      line0.set_data([], [])
      line1.set_data([], [])

      point1.set_data([], [])

      time_text.set_text('')
      return line0,line1, time_text, point1


  def animate(i):
      # Trajectory
      thisx0 = [0, np.sin(xx_star[0, i])]
      thisy0 = [0, np.cos(xx_star[0, i])]
      line0.set_data(thisx0, thisy0)

      # Reference
      thisx1 = [0, np.sin(xx_ref[0, -1])]
      thisy1 = [0, np.cos(xx_ref[0, -1])]
      line1.set_data(thisx1, thisy1)

      point1.set_data(i*dt, xx_star[0, i])

      time_text.set_text(time_template % (i*dt))
      return line0, line1, time_text, point1


  ani = animation.FuncAnimation(fig, animate, TT, interval=1, blit=True, init_func=init)
  ax.legend(loc="lower left")

  
  plt.show()


