# Linear Dynamics
# OPTCON 2022
# Lorenzo Sforni
# 20 Dec 2022
#

import numpy as np

def nominal_dynamics(xx, uu):
    """
        Double integrator NOMINAL dynamics
        xxp = [1, 1; 0, 1] xx + [0; 1] uu
    """

    ns = 2
    ni = 1

    xxp = np.zeros((ns,1))


    AAnom = np.array([[1,1],[0,1]])
    BBnom = np.array([[0],[1]])

    xxp = AAnom@xx + BBnom@uu

    fx = AAnom.T
    fu = BBnom.T

    return xxp, fx, fu

def real_dynamics(xx, uu):
    """
        Double integrator REAL dynamics
        xxp = AA xx + BB uu
    """

    ns = 2
    ni = 1

    xxp = np.zeros((ns,1))

    AA = np.array([[1,1],[0,1]])
    #AA = np.array([[1.1,0.8],[0,1.1]])
    BB = np.array([[0],[1]])

    xxp = AA@xx + BB@uu

    fx = AA.T
    fu = BB.T

    return xxp, fx, fu