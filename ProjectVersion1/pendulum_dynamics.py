#
# Optimal Control of a vehicle
# Discrete-time nonlinear dynamics of a pendulum (wfor testing othe code with a simpler dynamics)
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 04/01/2024
#

import numpy as np

# Constants for the dynamics
ns = 2  # number of states
ni = 1  # number of inputs
dt = 1e-3  # discretization stepsize - Forward Euler

# Pendulum parameters
mm = 0.4
ll = 0.2
kk = 1
gg = 9.81
KKeq = mm*gg*ll # K for equilibrium


#######################################
# Pendulum Dynamics
#######################################

def dynamics(xx, uu):
    """
    Nonlinear dynamics of a pendulum

    Args:
        xx (numpy.ndarray): State at time t, R^2.
        uu (numpy.ndarray): Input at time t, R^1.

    Returns:
        numpy.ndarray: Next state xx_{t+1}.
        numpy.ndarray: Gradient of f wrt xx, at xx,uu.
        numpy.ndarray: Gradient of f wrt uu, at xx,uu.
    """

    # Add a dimension for improving the compatibility of the code
    xx = xx[:, None]
    uu = uu[:, None]     
    
    # Preallocate the next state vector
    xxp = np.zeros((ns, 1))

    # Discrete-time nonlinear dynamics calculations for next state
    xxp[0] = xx[0,0] + dt * xx[1,0]
    xxp[1] = xx[1,0] + dt * (- gg / ll * np.sin(xx[0,0]) - kk / (mm * ll) * xx[1,0] + 1 / (mm * (ll ** 2)) * uu[0,0])                                       

    # Gradient computation (for future use in optimization)
    # Gradient wrt xx and uu (df/dx and df/du)
    fx = np.zeros((ns, ns))
    fu = np.zeros((ni, ns))

    # Derivative of dynamics w.r.t. state (fx)
    fx[0,0] = 1
    fx[1,0] = dt

    # Derivative of dynamics w.r.t. input (fu)
    fu[0,0] = 0

    # Removing singleton dimensions for the next state
    xxp = xxp.squeeze()

    return xxp, fx, fu