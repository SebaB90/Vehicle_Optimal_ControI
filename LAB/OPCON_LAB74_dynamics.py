#
# Gradient method for Optimal Control
# Discrete-time nonlinear dynamics
# Lorenzo Sforni
# Bologna, 22/11/2022
#

import numpy as np

ns = 2
ni = 1

dt = 1e-3 # discretization stepsize - Forward Euler

# Dynamics parameters

mm = 0.4
ll = 0.2
kk = 1
gg = 9.81

KKeq = mm*gg*ll # K for equilibrium

def dynamics(xx,uu):
  """
    Nonlinear dynamics of a pendulum

    Args
      - xx \in \R^2 state at time t
      - uu \in \R^1 input at time t

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
  
  """
  xx = xx[:,None]
  uu = uu[:,None]

  xxp = np.zeros((ns,1))

  xxp[0] = xx[0,0] + dt * xx[1,0]
  xxp[1] = xx[1,0] + dt * (- gg / ll * np.sin(xx[0,0]) - kk / (mm * ll) * xx[1,0] + 1 / (mm * (ll ** 2)) * uu[0,0])

  # Gradient

  fx = np.zeros((ns, ns))
  fu = np.zeros((ni, ns))

  #df1
  fx[0,0] = 1
  fx[1,0] = dt

  fu[0,0] = 0

  #df2

  fx[0,1] = dt*-gg / ll * np.cos(xx[0,0])
  fx[1,1] = 1 + dt*(- kk / (mm * ll))

  fu[0,1] = dt / (mm * (ll ** 2))

  xxp = xxp.squeeze()

  return xxp, fx, fu

