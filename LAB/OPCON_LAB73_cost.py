#
# Gradient method for Optimal Control
# Cost functions
# Lorenzo Sforni
# Bologna, 22/11/2022
#

import numpy as np

import OPCON_LAB74_dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# QQt = np.array([[10000, 0], [0, 100]])
QQt = 0.1*np.diag([100.0, 1.0])
RRt = 0.01*np.eye(ni)
# RRt = 1*np.eye(ni)

QQT = QQt

# ######################################
# # Reference curve
# ######################################

# ref_deg_T = 30
# KKeq = dyn.KKeq
# xx_ref_T = np.zeros((ns,))
# uu_ref_T = np.zeros((ni,))

# xx_ref_T[0] = np.deg2rad(ref_deg_T)
# uu_ref_T[0] = KKeq*np.sin(xx_ref_T[0])

# fx,fu = dyn.dynamics(xx_ref_T,uu_ref_T)[1:]

# AA = fx.T; BB = fu.T

# import control as ctrl

# QQT = ctrl.dare(AA,BB,QQt,RRt)[0]


def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  return ll.squeeze(), lx, lu

def termcost(xx,xx_ref):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

  lTx = QQT@(xx - xx_ref)

  return llT.squeeze(), lTx