#
# Optimal Control of aa vehicle
# Discrete-time nonlinear dynamics of aa vehicle
# Bertamè Sebastiano
# Bologna, 04/01/2024
#
from sympy import symbols, diff
import numpy as np
vehicle_dyn = True          # change to switch between bicycle or pendulum dynamics

dt = 1e-3        # discretization stepsize - Forward Euler
tf = 10          # Final time in seconds
TT = int(tf/dt)  # Number of discrete-time samples
TT_mid = TT/2

if vehicle_dyn:
    # Constants for the dynamics
    ns = 6  # number of states
    ni = 2  # number of inputs

    # Vehicle parameters
    mm = 1480  # Kg
    Iz = 1950  # Kg*mm^2
    aa = 1.421  # mm           
    bb = 1.029  # mm
    mi = 1  # nodim
    gg = 9.81  # mm/s^2

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
        Nonlinear dynamics of aa vehicle

        Args:
            xx (numpy.ndarray): State at time t, R^6.
            uu (numpy.ndarray): Input at time t, R^2.

        Returns:
            numpy.ndarray: Next state xx_{t+1}.
            numpy.ndarray: Gradient of f wrt xx, at xx,uu.
            numpy.ndarray: Gradient of f wrt uu, at xx,uu.
        """

        # Add aa dimension for improving the compatibility of the code
        xx = xx[:, None]
        uu = uu[:, None]     
        
        # Preallocate the next state vector
        xxp = np.zeros((ns,1))
        
        # Pre-compute repeated terms for efficiency
        cos_xx4 = np.cos(xx[4,0])
        sin_xx4 = np.sin(xx[4,0])
        cos_xx2 = np.cos(xx[2,0])
        sin_xx2 = np.sin(xx[2,0])
        cos_xx4_minus_uu0 = np.cos(xx[4,0] - uu[0,0])
        sin_xx4_minus_uu0 = np.sin(xx[4,0] - uu[0,0])

        # Compute slip angles for front and rear (Beta_f, Beta_r)
        Beta = ([
            uu[0,0] - (xx[3,0]*sin_xx4 + aa*xx[5,0]) / (xx[3,0]*cos_xx4), 
            - (xx[3,0]*sin_xx4 - bb*xx[5,0]) / (xx[3,0]*cos_xx4)
        ])

        dBetax3 = ([
            aa*xx[5,0]/((xx[3,0]**2)*cos_xx4),
            -(bb*xx[5,0])/((xx[3,0]**2)*cos_xx4)
        ])
        
        dBetax4 = ([
            -(xx[3,0] + sin_xx4*aa*xx[5,0])/(xx[3,0]*(cos_xx4**2)),
            -(xx[3,0] - sin_xx4*bb*xx[5,0])/(xx[3,0]*(cos_xx4**2))
        ])

        dBetax5 = ([
            -aa/(xx[3,0]*cos_xx4),
            bb/(xx[3,0]*cos_xx4)
        ])

        dBetau0 = ([
            1,
            0
        ])
        
        # Compute vertical forces at front and rear (F_zf, F_zr)
        Fz = ([mm*gg*bb/(aa+bb), mm*gg*aa/(aa+bb)])

        # Compute lateral forces at front and rear (F_yf, F_yr)
        Fy = ([mi*Fz[0]*Beta[0], mi*Fz[1]*Beta[1]])

        dFyx3 = ([mi*Fz[0]*dBetax3[0], mi*Fz[1]*dBetax3[1]])
        dFyx4 = ([mi*Fz[0]*dBetax4[0], mi*Fz[1]*dBetax4[1]])
        dFyx5 = ([mi*Fz[0]*dBetax5[0], mi*Fz[1]*dBetax5[1]])
        dFyu0 = ([mi*Fz[0]*dBetau0[0], mi*Fz[1]*dBetau0[1]])

        
        # Discrete-time nonlinear dynamics calculations for next state
        xxp[0] = xx[0,0] + dt * (xx[3,0] * cos_xx4 * cos_xx2 - xx[3,0] * sin_xx4 * sin_xx2)                                 
        xxp[1] = xx[1,0] + dt * (xx[3,0] * cos_xx4 * sin_xx2 + xx[3,0] * sin_xx4 * cos_xx2)                                 
        xxp[2] = xx[2,0] + dt * xx[5,0]                                                                                                           
        xxp[3] = xx[3,0] + dt * ((Fy[1] * sin_xx4 + uu[1,0] * cos_xx4_minus_uu0 + Fy[0] * sin_xx4_minus_uu0)/mm)                     
        xxp[4] = xx[4,0] + dt * ((Fy[1] * cos_xx4 + Fy[0] * cos_xx4_minus_uu0 - uu[1,0] * sin_xx4_minus_uu0)/(mm * xx[3,0]) - xx[5,0])   
        xxp[5] = xx[5,0] + dt * (((uu[1,0] * np.sin(uu[0,0]) + Fy[0] * np.cos(uu[0,0])) * aa - Fy[1] * bb)/Iz) 

        # Discrete-time nonlinear dynamics calculations for next state
       
        # Gradient computation (for future use in optimization)
        # Gradient wrt xx and uu (df/dx and df/du)
        fx = np.zeros((ns, ns))
        fu = np.zeros((ni, ns))
    
        # Derivative of dynamics w.r.t. state (fx)
        fx[0,:] = [1, 0, 0, 0, 0, 0]
        fx[1,:] = [0, 1, 0, 0, 0, 0]
        fx[2,:] = [dt*(-xx[3,0] * cos_xx4 * sin_xx2 - xx[3,0] * sin_xx4 * cos_xx2), dt*(xx[3,0] * cos_xx4 * cos_xx2 - xx[3,0] * sin_xx4 * sin_xx2), 1, 0, 0, 0]
        fx[3,:] = [dt*(cos_xx4 * cos_xx2 - sin_xx4 * sin_xx2), dt*(cos_xx4 * sin_xx2 + sin_xx4 * cos_xx2), 0, 1 + dt*((dFyx3[1]*sin_xx4 + dFyx3[0]*sin_xx4_minus_uu0)/mm), dt*(((dFyx3[1]*cos_xx4 + dFyx3[0]*cos_xx4_minus_uu0)*(mm * xx[3,0]) - mm*(Fy[1]*cos_xx4 + Fy[0]*cos_xx4_minus_uu0 - uu[1,0]*sin_xx4_minus_uu0))/((mm * xx[3,0])**2)), dt*(((dFyx3[0]*np.cos(uu[0,0]))*aa - dFyx3[1]*bb)/Iz)]
        fx[4,:] = [dt*(-xx[3,0]*sin_xx4*cos_xx2 - xx[3,0]*cos_xx4*sin_xx2), dt*(-xx[3,0]*sin_xx4*sin_xx2 + xx[3,0]*cos_xx4*cos_xx2), 0, dt*((dFyx4[1]*sin_xx4 + Fy[1]*cos_xx4 - uu[1,0]*sin_xx4_minus_uu0 + dFyx4[0]*sin_xx4_minus_uu0 + Fy[0]*cos_xx4_minus_uu0)/mm), 1 + dt*((dFyx4[1]*cos_xx4 - Fy[1]*sin_xx4 + dFyx4[0]*cos_xx4_minus_uu0 - Fy[0]*sin_xx4_minus_uu0 - uu[1,0]*cos_xx4_minus_uu0)/(mm*xx[3,0])), dt*(((dFyx4[0]*np.cos(uu[0,0]))*aa - dFyx4[1]*bb)/Iz)]
        fx[5,:] = [0, 0, dt, dt*((dFyx5[1]*sin_xx4 + dFyx5[0]*sin_xx4_minus_uu0)/mm), dt*((dFyx5[1]*cos_xx4 + dFyx5[0]*cos_xx4_minus_uu0)/(mm * xx[3,0]) - 1), 1 + dt*(((dFyx5[0]*np.cos(uu[0,0]))*aa - dFyx5[1]*bb)/Iz)]
        # Derivative of dynamics w.r.t. inputs (fu)
        fu[0,:] = [0, 0, 0, dt*((dFyu0[1]*sin_xx4 + uu[1,0]*sin_xx4_minus_uu0 + dFyu0[0]*sin_xx4_minus_uu0 - Fy[0]*cos_xx4_minus_uu0)/mm), dt*((dFyu0[1]*cos_xx4 + dFyu0[0]*cos_xx4_minus_uu0 + Fy[0]*sin_xx4_minus_uu0 + uu[1,0]*cos_xx4_minus_uu0)/(mm * xx[3,0])),  dt*(((uu[1,0]*np.cos(uu[0,0]) + dFyu0[0]*np.cos(uu[0,0]) - Fy[0]*np.sin(uu[0,0]))*aa - dFyu0[1]*bb)/Iz)]
        fu[1,:] = [0, 0, 0, dt*cos_xx4_minus_uu0/mm, dt*(- sin_xx4_minus_uu0)/(mm * xx[3,0]), dt*np.sin(uu[0,0])*aa/Iz]
        '''
        xxp[0] = xx[0,0] + dt * (xx[3,0] * np.cos(xx[4,0]) * np.cos(xx[2,0]) - xx[3,0] * np.sin(xx[4,0]) * np.sin(xx[2,0]))
        xxp[1] = xx[1,0] + dt * (xx[3,0] * np.cos(xx[4,0]) * np.sin(xx[2,0]) + xx[3,0] * np.sin(xx[4,0]) * np.cos(xx[2,0]))
        xxp[2] = xx[2,0] + dt * xx[5,0]
        xxp[3] = xx[3,0] + dt * ((mi*(mm*gg*aa/(aa+bb))*(- (xx[3,0]*np.sin(xx[4,0]) - bb*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * np.sin(xx[4,0]) + uu[1,0] * np.cos(xx[4,0] - uu[0,0]) + mi*(mm*gg*bb/(aa+bb))*(uu[0,0] - (xx[3,0]*np.sin(xx[4,0]) + aa*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * np.sin(xx[4,0] - uu[0,0]))/mm)      
        xxp[4] = xx[4,0] + dt * ((mi*(mm*gg*aa/(aa+bb))*(- (xx[3,0]*np.sin(xx[4,0]) - bb*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * np.cos(xx[4,0]) + mi*(mm*gg*bb/(aa+bb))*(uu[0,0] - (xx[3,0]*np.sin(xx[4,0]) + aa*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * np.cos(xx[4,0] - uu[0,0]) - uu[1,0] * np.sin(xx[4,0] - uu[0,0]))/(mm * xx[3,0]) - xx[5,0]) 
        xxp[5] = xx[5,0] + dt * (((uu[1,0] * np.sin(uu[0,0]) + mi*(mm*gg*bb/(aa+bb))*(uu[0,0] - (xx[3,0]*np.sin(xx[4,0]) + aa*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * np.cos(uu[0,0])) * aa - mi*(mm*gg*aa/(aa+bb))*(- (xx[3,0]*np.sin(xx[4,0]) - bb*xx[5,0]) / (xx[3,0]*np.cos(xx[4,0]))) * bb)/Iz)
        
        # Gradient computation (for future use in optimization)
        # Gradient wrt xx and uu (df/dx and df/du)
        fx = np.zeros((ns, ns))
        fu = np.zeros((ni, ns))

        # Derivative of dynamics w.r.t. state (fx)
        fx[0,:] = [1, 0, 0, 0, 0, 0]
        fx[1,:] = [0, 1, 0, 0, 0, 0]
        fx[2,:] = [dt*(-xx[3,0]*np.sin(xx[2,0])*np.cos(xx[4,0]) - xx[3,0]*np.sin(xx[4,0])*np.cos(xx[2,0])), dt*(-xx[3,0]*np.sin(xx[2,0])*np.sin(xx[4,0]) + xx[3,0]*np.cos(xx[2,0])*np.cos(xx[4,0])), 1, 0, 0, 0]
        fx[3,:] = [dt*(-np.sin(xx[2,0])*np.sin(xx[4,0]) + np.cos(xx[2,0])*np.cos(xx[4,0])), dt*(np.sin(xx[2,0])*np.cos(xx[4,0]) + np.sin(xx[4,0])*np.cos(xx[2,0])), 0, dt*(-aa*gg*mi*mm*np.sin(xx[4,0])**2/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])) - aa*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])/(xx[3,0]**2*(aa + bb)*np.cos(xx[4,0])) - bb*gg*mi*mm*(-np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) + (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0])))*np.sin(uu[0,0] - xx[4,0])/(aa + bb))/mm + 1, dt*((-aa*gg*mi*mm*np.sin(xx[4,0])/(xx[3,0]*(aa + bb)) - aa*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*(aa + bb)) + bb*gg*mi*mm*(-np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) + (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0])))*np.cos(uu[0,0] - xx[4,0])/(aa + bb))/(mm*xx[3,0]) - (aa*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*(aa + bb)) + bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.cos(uu[0,0] - xx[4,0])/(aa + bb) + uu[1,0]*np.sin(uu[0,0] - xx[4,0]))/(mm*xx[3,0]**2)), dt*(aa*bb*gg*mi*mm*(-np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) + (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0])))*np.cos(uu[0,0])/(aa + bb) + aa*bb*gg*mi*mm*np.sin(xx[4,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])) + aa*bb*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*(aa + bb)*np.cos(xx[4,0])))/Iz]
        fx[4,:] = [dt*(-xx[3,0]*np.sin(xx[2,0])*np.cos(xx[4,0]) - xx[3,0]*np.sin(xx[4,0])*np.cos(xx[2,0])), dt*(-xx[3,0]*np.sin(xx[2,0])*np.sin(xx[4,0]) + xx[3,0]*np.cos(xx[2,0])*np.cos(xx[4,0])), 0, dt*(-aa*gg*mi*mm*np.sin(xx[4,0])/(aa + bb) + aa*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])**2/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])**2) + aa*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*(aa + bb)) - bb*gg*mi*mm*(-1 - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])**2))*np.sin(uu[0,0] - xx[4,0])/(aa + bb) + bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.cos(uu[0,0] - xx[4,0])/(aa + bb) + uu[1,0]*np.sin(uu[0,0] - xx[4,0]))/mm, dt*(-aa*gg*mi*mm*np.cos(xx[4,0])/(aa + bb) + bb*gg*mi*mm*(-1 - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])**2))*np.cos(uu[0,0] - xx[4,0])/(aa + bb) + bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.sin(uu[0,0] - xx[4,0])/(aa + bb) - uu[1,0]*np.cos(uu[0,0] - xx[4,0]))/(mm*xx[3,0]) + 1, dt*(aa*bb*gg*mi*mm*(-1 - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])**2))*np.cos(uu[0,0])/(aa + bb) + aa*bb*gg*mi*mm/(aa + bb) - aa*bb*gg*mi*mm*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0]))*np.sin(xx[4,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])**2))/Iz]
        fx[5,:] = [0, 0, dt, dt*(aa*bb*gg*mi*mm*np.sin(xx[4,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])) + aa*bb*gg*mi*mm*np.sin(uu[0,0] - xx[4,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])))/mm, dt*(-1 + (aa*bb*gg*mi*mm/(xx[3,0]*(aa + bb)) - aa*bb*gg*mi*mm*np.cos(uu[0,0] - xx[4,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])))/(mm*xx[3,0])), 1 + dt*(-aa**2*bb*gg*mi*mm*np.cos(uu[0,0])/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])) - aa*bb**2*gg*mi*mm/(xx[3,0]*(aa + bb)*np.cos(xx[4,0])))/Iz]
        # Derivative of dynamics w.r.t. inputs (fu)
        fu[0,:] = [0, 0, 0, dt*(-bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.cos(uu[0,0] - xx[4,0])/(aa + bb) - bb*gg*mi*mm*np.sin(uu[0,0] - xx[4,0])/(aa + bb) - uu[1,0]*np.sin(uu[0,0] - xx[4,0]))/mm, dt*(-bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.sin(uu[0,0] - xx[4,0])/(aa + bb) + bb*gg*mi*mm*np.cos(uu[0,0] - xx[4,0])/(aa + bb) + uu[1,0]*np.cos(uu[0,0] - xx[4,0]))/(mm*xx[3,0]), aa*dt*(-bb*gg*mi*mm*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))*np.sin(uu[0,0])/(aa + bb) + bb*gg*mi*mm*np.cos(uu[0,0])/(aa + bb) + uu[1,0]*np.cos(uu[0,0]))/Iz]
        fu[1,:] = [0, 0, 0, dt*np.cos(uu[0,0] - xx[4,0])/mm, dt*np.sin(uu[0,0] - xx[4,0])/(mm*xx[3,0]), aa*dt*np.sin(uu[0,0])/Iz]    
        '''
        # Removing singleton dimensions for the next state
        xxp = xxp.squeeze()


        ############################################## PENDULUM DYNAMICS ########################################################

    else:
        """
        Nonlinear dynamics of aa pendulum

        Args:
            xx (numpy.ndarray): State at time t, R^2.
            uu (numpy.ndarray): Input at time t, R^1.

        Returns:
            numpy.ndarray: Next state xx_{t+1}.
            numpy.ndarray: Gradient of f wrt xx, at xx,uu.
            numpy.ndarray: Gradient of f wrt uu, at xx,uu.
        """

        # Add aa dimension for improving the compatibility of the code
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
