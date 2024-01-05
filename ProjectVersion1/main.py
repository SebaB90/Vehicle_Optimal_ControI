#
# Optimal Control of a vehicle
# Main
# Antonio Rapallini & Sebastiano Bertamé
# Bologna, 04/01/2024
#


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

# ATTENTION! Import the dynamics that you want to simulate, COMMENT THE OTHER! Do that also for newtons and costs files
#import vehicle_dynamics as dyn
import pendulum_dynamics as dyn

# import cost functions
import costs as cst

import newton as nwtn

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


############################################################
# Algorithm parameters
############################################################

max_iters = int(10)
stepsize_0 = 1

# ARMIJO PARAMETERS
cc = 0.5
beta = 0.7
armijo_maxiters = 20  # number of Armijo iterations

term_cond = 1e-6  # Termination condition

visu_armijo = False  # Visualize Armijo iterations
visu_animation = False  # Visualize trajectory animation

test = True # Set true for testing the open loop dynamics and the correctness of the derivatives


############################################################
# Trajectory parameters
############################################################

tf = 10  # Final time in seconds

dt = dyn.dt  # Get discretization step from dynamics
ns = dyn.ns  # Get the number of states from the dynamics
ni = dyn.ni  # Get the number of input from the dynamics

TT = int(tf/dt)  # Number of discrete-time samples
TT_mid = TT/2


############################################################
# Arrays to store data
############################################################

xx = np.zeros((ns, TT, max_iters))  # state seq.
uu = np.zeros((ni, TT, max_iters))  # input seq.

lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters))  # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))  # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)  # collect cost
descent = np.zeros(max_iters)  # collect descent direction
descent_arm = np.zeros(max_iters)  # collect descent direction


############################################################
# TESTS
############################################################

if test == False and ns==6:
    
    ######################################
    # OPEN LOOP TEST TO CHECK IF THE VEHICLE DYNAMICS IS CORRECT
    ######################################

    # Defining an x and a u
    uu = np.array([0.25, 20])
    xx = np.array([0, 0, 0, 1, 0, 0])

    # Variables for evaluating the dynamics in time
    x_traj = [np.copy(xx[0])]
    y_traj = [np.copy(xx[1])]
    traj = np.copy(xx)

    total_time = 100  # Adjust the total simulation time as needed
    num_steps = int(total_time / dt)

    for i in range(num_steps-1):
        traj = dyn.dynamics(traj, uu)[0]
        x_traj.append(traj[0])
        y_traj.append(traj[1])

    # Plotting the obtained trajectory
    fig, axs = plt.subplots(2, 1, sharex='all')

    axs[0].plot(np.linspace(0, TT, num_steps), x_traj, 'g', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x$')
    axs[0].set_title('X Trajectory')  # Add a title to the first subplot


    axs[1].plot(np.linspace(0, TT, num_steps), y_traj, 'g', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y$')
    axs[1].set_xlabel('time')
    axs[1].set_title('Y Trajectory')  # Add a title to the first subplot


    fig.align_ylabels(axs)
    plt.show()

    # Plot of the x-y trajectory
    plt.plot(x_traj, y_traj, 'g', linewidth=2)
    plt.title('Vehicle Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

    ######################################
    # CHECK OF THE DERIVATIVES
    ######################################
    '''
    As simple test we check if:
        f(x_bar + dltx) - f(x_bar) = df/dx(evaluated in x_bar) * dltx
    '''

    # Define the infinitesimal increments for evaluating the derivatives
    dx = 1e-3 
    du = 1e-3       
    
    # Evaluated the derivatives from the dynamics
    xxp, fx, fu = dyn.dynamics(xx, uu)
    
    # Evaluate the A and B matrices
    AA = fx.T
    BB = fu.T
    
    # Arrays to store data
    xdx = np.zeros((ns,))
    dltx = np.zeros((ns,))
    udu = np.zeros((ni,))
    dltu = np.zeros((ni,))
    
    for i in range (0,ns):
        dltx[i] = dx

    for i in range (0,ni):
        dltu[i] = du

    xdx = xx + dltx
    xx_plus = dyn.dynamics(xdx, uu)[0]
    diff_x = xx_plus - xxp
    check_x = diff_x - np.dot(AA,dltx)

    udu = uu + dltu
    xx_plus = dyn.dynamics(xx, udu)[0]    
    diff_u = xx_plus - xxp     
    check_u = diff_u - np.dot(BB,dltu)

    print("\n\n")
    blue_bold_title = "\033[1;34mERROR IN THE EVALUATED DERIVATIVES:\033[0m"
    print(blue_bold_title)
    print(f'\nError in derivatives of x is:\n{check_x}')
    print(f'\nError in derivatives of u is:\n{check_u}\n')


########################################################################
############## TASK 1: TRAJECTORY GENERATION (I) #######################
########################################################################

if ns == 6:
    
    # Import of the parameters of the system
    mm = dyn.mm  # Kg
    Iz = dyn.Iz  # Kg*m^2
    aa = dyn.aa  # m           
    bb = dyn.bb  # m
    mi = dyn.mi  # nodim
    gg = dyn.gg  # m/s^2

    eq = np.zeros((ns+ni, 2))
    xx_eq = np.zeros((ns,2))
    uu_eq = np.zeros((ni,2))


    ############################################################
    # Evalutaion of two equilibria for the system
    ############################################################
    
    # We have evaluated the cornering equilibria, so setting Betadot = 0, Vdot = 0 and Psidotdot = 0
    # Then imposing V(x3) and Beta(x4) we evaluate the other states and inputs

    def equations(vars):
        x4, u0, u1 = vars
        Beta = [u0 - (x3 * np.sin(x4) + aa * x5) / (x3 * np.cos(x4)),
                - (x3 * np.sin(x4) - bb * x5) / (x3 * np.cos(x4))]  # Beta = [Beta_f, Beta_r]
        Fz = [mm * gg * bb / (aa + bb),
            mm * gg * aa / (aa + bb)]  # Fz = [F_zf, F_zr]
        Fy = [mi * Fz[0] * Beta[0],
            mi * Fz[1] * Beta[1]]  # Fy = [F_yf, F_yr]

        eq1 = (Fy[1] * np.sin(x4) + u1 * np.cos(x4 - u0) + Fy[0] * np.sin(x4 - u0)) / mm  # V dot (x3)
        eq2 = (Fy[1] * np.cos(x4) + Fy[0] * np.cos(x4 - u0) - u1 * np.sin(x4 - u0)) / (mm * x3) - x5  # Beta dot (x4)
        eq3 = ((u1 * np.sin(u0) + Fy[0] * np.cos(u0)) * aa - Fy[1] * bb) / Iz  # Psi dot dot (x5)

        return [eq1, eq2, eq3]

    # Initial guess for the fsolve evaluation
    initial_guess = [0.5, 0.1, 0.1]  # [x5(0), u0(0), u1(0)]

    #######################
    # FIRST EQUILIBRIUM
    #######################

    # Imposing x3 and x4 we evaluate the other parameters
    x3 = 3                 
    x5 = 0
    eq[3,0] = np.copy(x3)  # V
    eq[5,0] = np.copy(x5)  # psi 
    # Using fsolve we evaluate psi dot, steering angle, force
    eq[4,0] = fsolve(equations, initial_guess)[0]  # beta
    eq[6:,0] = fsolve(equations, initial_guess)[1:]  # steering angle, force
    # We evaluate x, y and psi by integration
    eq[2,0] = eq[5,0]*TT_mid # psi   
    eq[0,0] = (eq[3,0]*np.cos(eq[4,0])*np.cos(eq[2,0])-eq[3,0]*np.sin(eq[4,0])*np.sin(eq[2,0]))*TT_mid  # x
    eq[1,0] = (eq[3,0]*np.cos(eq[4,0])*np.sin(eq[2,0])+eq[3,0]*np.sin(eq[4,0])*np.cos(eq[2,0]))*TT_mid  # y

    #######################
    # SECOND EQUILIBRIUM
    #######################
    x3 = 5                 
    x5 = 0.2
    eq[3,1] = np.copy(x3)  # V
    eq[5,1] = np.copy(x5)  # psi 
    # Using fsolve we evaluate psi dot, steering angle, force
    eq[4,1] = fsolve(equations, initial_guess)[0]  # beta
    eq[6:,1] = fsolve(equations, initial_guess)[1:]  # steering angle, force
    # We evaluate x, y and psi by integration
    eq[2,1] = eq[5,1]*TT_mid # psi   
    eq[0,1] = (eq[3,1]*np.cos(eq[4,1])*np.cos(eq[2,1])-eq[3,1]*np.sin(eq[4,1])*np.sin(eq[2,1]))*TT_mid  # x
    eq[1,1] = (eq[3,1]*np.cos(eq[4,1])*np.sin(eq[2,1])+eq[3,1]*np.sin(eq[4,1])*np.cos(eq[2,1]))*TT_mid  # y

    xx_eq = eq[:ns]
    uu_eq = eq[ns:]

    # Print the result
    print("\n\n")
    blue_bold_title = "\033[1;34mEVALUATED EQUILIBRIUM:\033[0m"
    print(blue_bold_title)
    print(f" \nxx at Equilibrium 1:\n  {xx_eq[0:, 0]}")
    print(f" \nuu at Equilibrium 1:\n  {uu_eq[0:, 0]}")
    print(f" \nxx at Equilibrium 2:\n  {xx_eq[0:, 1]}")
    print(f" \nuu at Equilibrium 2:\n  {uu_eq[0:, 1]}\n\n")
    
    
    ############################################################
    # Evalutaion of the reference trajectory
    ############################################################

    traj_ref = np.zeros((ns+ni, TT))
    traj_ref[3:,0] = eq[3:,0]
    
    xx_ref = traj_ref[0:6,:]
    uu_ref = traj_ref[6:,:]

    # Step reference signal - for all the states

    for tt in range(1,TT):
    
        traj = dyn.dynamics(traj_ref[:6,tt-1], traj_ref[6:,tt-1])[0]
        traj_ref[:3, tt] = traj[:3]

        if tt < TT_mid:
            traj_ref[3:, tt] = eq[3:,0]

        else:  
            traj_ref[3:, tt] = eq[3:,1]

    # Plot of the reference trajcetory
    tt_hor = np.linspace(0,tf,TT)

    ####################################################################################################################################
    ### Plot to test trajectory reference
    plt.plot(traj_ref[0,:], traj_ref[1,:], label='Trajectory')
    plt.title('Vehicle Reference Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()
    #######################################################################################################################################

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, traj_ref[0,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x$')

    axs[1].plot(tt_hor, traj_ref[1,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y$')

    axs[2].plot(tt_hor, traj_ref[2,:], 'g--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$psi$')

    axs[3].plot(tt_hor, traj_ref[3,:], 'g--', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$V$')

    axs[4].plot(tt_hor, traj_ref[4,:], 'g--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$beta$')

    axs[5].plot(tt_hor, traj_ref[5,:], 'g--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$psi dot$')

    axs[6].plot(tt_hor, traj_ref[6,:], 'g--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$u_0$')

    axs[7].plot(tt_hor, traj_ref[7,:], 'g--', linewidth=2)
    axs[7].grid()
    axs[7].set_ylabel('$u_1$')
    axs[7].set_xlabel('time')

    fig.suptitle("Reference", fontsize=16)
    fig.align_ylabels(axs)

    plt.show()
    
    

if ns == 2:
    ######################################
    # Reference curve for the pendulum
    ######################################
    ref_deg_0 = 0
    ref_deg_T = 30

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Step reference
    KKeq = dyn.KKeq
    xx_ref[0, int(TT/2):] = np.ones((1, int(TT/2))) * np.deg2rad(ref_deg_T)
    uu_ref[0, :] = KKeq * np.sin(xx_ref[0, :])

    x0 = xx_ref[:, 0]
    
    # Plot of the reference trajcetory
    tt_hor = np.linspace(0,tf,TT)

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x_1$')

    axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$x_2$')

    axs[2].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$u$')
    axs[2].set_xlabel('time')
  
    plt.show()



############################################################
# Initial guess
############################################################

xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))


# perform Newton's like method

xx, uu, descent, JJ = nwtn.Newton(xx_ref, uu_ref, max_iters)

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2]        # for plotting purposes

# Plots

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

# Design OPTIMAL TRAJECTORY  ---------------------------------------------------------------------------------------

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

if ns == 6:
    axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
    axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x$')

    axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
    axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y$')

    axs[2].plot(tt_hor, xx_star[2,:],'r', linewidth=2)
    axs[2].plot(tt_hor, xx_ref[2,:], 'r--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$psi$')

    axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
    axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$V$')

    axs[4].plot(tt_hor, xx_star[4,:], linewidth=2)
    axs[4].plot(tt_hor, xx_ref[4,:], 'g--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$beta$')

    axs[5].plot(tt_hor, xx_star[5,:],'r', linewidth=2)
    axs[5].plot(tt_hor, xx_ref[5,:], 'r--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$psi dot$')

    axs[6].plot(tt_hor, uu_star[0,:], linewidth=2)
    axs[6].plot(tt_hor, uu_ref[0,:], 'g--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$delta$')

    axs[7].plot(tt_hor, uu_star[1,:],'r', linewidth=2)
    axs[7].plot(tt_hor, uu_ref[1,:], 'r--', linewidth=2)
    axs[7].grid()
    axs[7].set_ylabel('$F$')
    axs[7].set_xlabel('time')
    
if ns == 2:
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

# Plotting the trajectory
plt.plot(xx_star[0,:], xx_star[1,:], label='Trajectory')
plt.title('Vehicle Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()


########################################################################
############## TASK 1: TRAJECTORY GENERATION (II) #######################
########################################################################

############################################################
# SMOOTHING the reference trajectory
############################################################

if ns == 6:
    
    tt_hor = range(TT)

    # Perform linear interpolation for reference trajectory
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')
    fig.suptitle('Trajectory Smoothing using PCHIP Spline')
    traj_smooth = np.zeros((ns+ni,TT))
    x_traj_smooth = np.zeros((ns+ni,TT))

    axs[0].plot(tt_hor, traj_ref[0, :], 'g--', linewidth=2, label='Original Reference Trajectory')
    axs[0].grid()
    axs[1].plot(tt_hor, traj_ref[1, :], 'g--', linewidth=2, label='Original Reference Trajectory')
    axs[1].grid()
    axs[2].plot(tt_hor, traj_ref[2, :], 'g--', linewidth=2, label='Original Reference Trajectory')
    axs[2].grid()

    for i in range (3,ns+ni):
        new_num_points = 7      # Adjust the number of points for a smoother curve
        interp_indices = np.linspace(0, TT - 1, new_num_points)
        new_traj_ref_0 = np.interp(interp_indices, tt_hor, traj_ref[i,:])

        # define point to create spline
        x_spl = np.array([interp_indices[0], interp_indices[1], interp_indices[2], interp_indices[4], interp_indices[5], interp_indices[6]])
        y_spl = np.array([new_traj_ref_0[0], new_traj_ref_0[1], new_traj_ref_0[2], new_traj_ref_0[4], new_traj_ref_0[5], new_traj_ref_0[6]])

        # Create a piecewise cubic Hermite interpolating polynomial(PCHIP) interpolation of the given points
        cs = PchipInterpolator(x_spl, y_spl)

        # Generate new, smoother x values (denser for plotting)
        x_spl_new = np.linspace(min(x_spl), max(x_spl), TT)

        # Compute the smoothed y values
        y_spl_new = cs(x_spl_new)

        # Store the values inside an array
        traj_smooth[i,:] = y_spl_new

        # Plotting the original and smoothed trajectories
        axs[i].plot(tt_hor, traj_ref[i, :], 'g--', linewidth=2, label='Original Reference Trajectory')
        axs[i].plot(interp_indices, new_traj_ref_0, 'b--', linewidth=2, label='Interpolated Trajectory')
        axs[i].plot(x_spl, y_spl, 'o', label='Points used for spline creation')
        axs[i].plot(x_spl_new, y_spl_new, 'r-', label='Smoothed Trajectory')
        axs[i].grid()
        if i == ns+ni:
            axs[i].xlabel('time')

    axs[0].set_ylabel('$x$')
    axs[1].set_ylabel('$y$')
    axs[2].set_ylabel('$psi$')
    axs[3].set_ylabel('$V$')
    axs[4].set_ylabel('$beta$')
    axs[5].set_ylabel('$psi dot$')
    axs[6].set_ylabel('$delta$')
    axs[7].set_ylabel('$F$')

    plt.legend()
    plt.show()



