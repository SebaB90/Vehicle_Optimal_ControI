#
# Optimal Control of a vehicle
# Discrete-time nonlinear dynamics of a vehicle
# Bertamè Sebastiano
# Bologna, 04/01/2024
#

import numpy as np
vehicle_dyn = False          # change to switch between bicycle or pendulum dynamics

dt = 1e-3  # discretization stepsize - Forward Euler
tf = 10  # Final time in seconds
TT = int(tf/dt)  # Number of discrete-time samples
TT_mid = TT/2

if vehicle_dyn:
    # Constants for the dynamics
    ns = 6  # number of states
    ni = 2  # number of inputs

    # Vehicle parameters
    mm = 1480  # Kg
    Iz = 1950  # Kg*m^2
    aa = 1.421  # m           
    bb = 1.029  # m
    mi = 1  # nodim
    gg = 9.81  # m/s^2

else:
    # Constants for the dynamics
    ns = 2  # number of states
    ni = 1  # number of inputs

    # Pendulum parameters
    mm = 0.4
    ll = 0.2
    kk = 1
    gg = 9.81
    KKeq = mm*gg*ll # K for equilibrium



#######################################
# Car Dynamics
#######################################

def dynamics(xx, uu):

    ############################################## BICYCLE DYNAMICS ########################################################

    if vehicle_dyn:
        """
        Nonlinear dynamics of a vehicle

        Args:
            xx (numpy.ndarray): State at time t, R^6.
            uu (numpy.ndarray): Input at time t, R^2.

        Returns:
            numpy.ndarray: Next state xx_{t+1}.
            numpy.ndarray: Gradient of f wrt xx, at xx,uu.
            numpy.ndarray: Gradient of f wrt uu, at xx,uu.
        """

        # Add a dimension for improving the compatibility of the code
        # xx = xx[:, None]
        # uu = uu[:, None]     
        
        # Preallocate the next state vector
        xxp = np.zeros((ns,1))
        
        # Pre-compute repeated terms for efficiency
        cos_xx4 = np.cos(xx[4])
        sin_xx4 = np.sin(xx[4])
        cos_xx4_minus_uu0 = np.cos(xx[4] - uu[0])
        sin_xx4_minus_uu0 = np.sin(xx[4] - uu[0])
        m_xx3_sq = mm * (xx[3]**2)

        # Compute slip angles for front and rear (Beta_f, Beta_r)
        Beta = ([
            uu[0] - (xx[3]*sin_xx4 + aa*xx[5]) / (xx[3]*cos_xx4), 
            - (xx[3]*sin_xx4 - bb*xx[5]) / (xx[3]*cos_xx4)
        ])
        
        # Compute vertical forces at front and rear (F_zf, F_zr)
        Fz = ([mm*gg*bb/(aa+bb), mm*gg*aa/(aa+bb)])
        
        # Compute lateral forces at front and rear (F_yf, F_yr)
        Fy = ([mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]])

        # Discrete-time nonlinear dynamics calculations for next state
        xxp[0] = xx[0] + dt * (xx[3] * cos_xx4 * np.cos(xx[2]) - xx[3] * sin_xx4 * np.sin(xx[2]))                                 
        xxp[1] = xx[1] + dt * (xx[3] * cos_xx4 * np.sin(xx[2]) + xx[3] * sin_xx4 * np.cos(xx[2]))                                 
        xxp[2] = xx[2] + dt * xx[5]                                                                                                           
        xxp[3] = xx[3] + dt * ((Fy[1] * sin_xx4 + uu[1] * cos_xx4_minus_uu0 + Fy[0] * sin_xx4_minus_uu0)/mm)                     
        xxp[4] = xx[4] + dt * ((Fy[1] * cos_xx4 + Fy[0] * cos_xx4_minus_uu0 - uu[1] * sin_xx4_minus_uu0)/(mm * xx[3]) - xx[5])   
        xxp[5] = xx[5] + dt * (((uu[1] * np.sin(uu[0]) + Fy[0] * np.cos(uu[0])) * aa - Fy[1] * bb)/Iz)                                          

        # Gradient computation (for future use in optimization)
        # Gradient wrt xx and uu (df/dx and df/du)
        fx = np.zeros((ns, ns))
        fu = np.zeros((ni, ns))
    

        # Derivative of dynamics w.r.t. state (fx)
        fx = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [dt*(-xx[3] * cos_xx4 * np.sin(xx[2]) - xx[3] * sin_xx4 * np.cos(xx[2])), 
            dt*(xx[3] * cos_xx4 * np.cos(xx[2]) - xx[3] * sin_xx4 * np.sin(xx[2])), 1, 0, 0, 0],
            [dt*(cos_xx4 * np.cos(xx[2]) - sin_xx4 * np.sin(xx[2])), 
            dt*(cos_xx4 * np.sin(xx[2]) + sin_xx4 * np.cos(xx[2])), 0, 1, 
            dt*((Fy[1] * cos_xx4 + Fy[0] * cos_xx4_minus_uu0 - uu[1] * sin_xx4_minus_uu0)*(-1/m_xx3_sq)), 0],
            [dt*(-xx[3] * sin_xx4 * np.cos(xx[2]) - xx[3] * cos_xx4 * np.sin(xx[2])), 
            dt*(-xx[3] * sin_xx4 * np.sin(xx[2]) + xx[3] * cos_xx4 * np.cos(xx[2])), 0, 
            dt*((Fy[1] * cos_xx4 - uu[1] * sin_xx4_minus_uu0 + Fy[0] * cos_xx4_minus_uu0)/mm), 
            1 + dt*((-Fy[1] * sin_xx4 - Fy[0] * sin_xx4_minus_uu0 - uu[1] * cos_xx4_minus_uu0)/(mm * xx[3])), 0],
            [0, 0, dt, 0, -dt, 1]
        ])
        
        fu = np.array([[0, 0, 0, dt*(uu[1] * sin_xx4_minus_uu0 - Fy[0] * cos_xx4_minus_uu0)/mm, dt*(Fy[0] * sin_xx4_minus_uu0 + uu[1] * cos_xx4_minus_uu0)/(mm * xx[3]),  dt*((uu[1] * np.cos(uu[0]) - Fy[0] * np.sin(uu[0])) *aa/Iz)],
                [0, 0, 0, dt*cos_xx4_minus_uu0/mm, dt*(- sin_xx4_minus_uu0)/(mm * xx[3]), dt * np.sin(uu[0]) * aa/Iz]])
        # Removing singleton dimensions for the next state
        xxp = xxp.squeeze()


        ############################################## PENDULUM DYNAMICS ########################################################

    else:
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

        #df2
        fx[0,1] = dt*-gg / ll * np.cos(xx[0,0])
        fx[1,1] = 1 + dt*(- kk / (mm * ll))

        fu[0,1] = dt / (mm * (ll ** 2))

        # Removing singleton dimensions for the next state
        xxp = xxp.squeeze()
       
    return xxp, fx, fu
