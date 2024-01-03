#
# Discrete-time LQR solvers
# Lorenzo Sforni
# Bologna, 21/11/2022
#

import numpy as np


def lti_LQR(AA, BB, QQ, RR, QQf, TT):

  """
	LQR for LTV system with (time-varying) cost	
	
  Args
    - AA (nn x nn (x TT)) matrix
    - BB (nn x mm (x TT)) matrix
    - QQ (nn x nn (x TT)), RR (mm x mm (x TT)) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AA.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AA = AA[:,:,None]
    ns, lA = AA.shape[1:]

  try:  
    nu, lB = BB.shape[1:]
  except:
    BB = BB[:,:,None]
    ni, lB = BB.shape[1:]

  try:
      nQ, lQ = QQ.shape[1:]
  except:
      QQ = QQ[:,:,None]
      nQ, lQ = QQ.shape[1:]

  try:
      nR, lR = RR.shape[1:]
  except:
      RR = RR[:,:,None]
      nR, lR = RR.shape[1:]

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()


  if lA < TT:
      AA = AA.repeat(TT, axis=2)
  if lB < TT:
      BB = BB.repeat(TT, axis=2)
  if lQ < TT:
      QQ = QQ.repeat(TT, axis=2)
  if lR < TT:
      RR = RR.repeat(TT, axis=2)
  
  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = AAt.T@PPtp@AAt - AAt.T@PPtp@BBt@(RRt + BBt.T@PPtp@BBt)**-1 @BBt.T@PPtp@AAt + QQt 
  
  # Evaluate KK
  
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    KK[:,:,tt] = -(RRt+ (BBt.T@PPtp@BBt))**-1 *BBt.T@PPtp@AAt

  return KK, PP
    