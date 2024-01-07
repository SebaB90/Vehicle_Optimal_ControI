
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, cos, sin, diff

ns = 6
ni = 2

x0, x1, x2, x3, x4, x5, u0, u1, mi, mm, gg, aa, bb, Iz, dt = symbols('x0 x1 x2 x3 x4 x5 u0 u1 mi mm gg aa bb Iz dt')
xxp = [x0 + dt * (x3 * cos(x4) * cos(x2) - x3 * sin(x4) * sin(x2)),
       x1 + dt * (x3 * cos(x4) * sin(x2) + x3 * sin(x4) * cos(x2)),
       x2 + dt * x5,
       x3 + dt * ((mi*(mm*gg*aa/(aa+bb))*(- (x3*sin(x4) - bb*x5) / (x3*cos(x4))) * sin(x4) + u1 * cos(x4 - u0) + mi*(mm*gg*bb/(aa+bb))*(u0 - (x3*sin(x4) + aa*x5) / (x3*cos(x4))) * sin(x4 - u0))/mm),                     
       x4 + dt * ((mi*(mm*gg*aa/(aa+bb))*(- (x3*sin(x4) - bb*x5) / (x3*cos(x4))) * cos(x4) + mi*(mm*gg*bb/(aa+bb))*(u0 - (x3*sin(x4) + aa*x5) / (x3*cos(x4))) * cos(x4 - u0) - u1 * sin(x4 - u0))/(mm * x3) - x5),  
       x5 + dt * (((u1 * sin(u0) + mi*(mm*gg*bb/(aa+bb))*(u0 - (x3*sin(x4) + aa*x5) / (x3*cos(x4))) * cos(u0)) * aa - mi*(mm*gg*aa/(aa+bb))*(- (x3*sin(x4) - bb*x5) / (x3*cos(x4))) * bb)/Iz) 
]

# Discrete-time nonlinear dynamics calculations for next state
# Gradient computation (for future use in optimization)
# Gradient wrt xx and uu (df/dx and df/du)

# Derivative of dynamics w.r.t. state (fx)


fx0 = [diff(xxp[0],x0), diff(xxp[1],x0), diff(xxp[2],x0), diff(xxp[3],x0), diff(xxp[4],x0), diff(xxp[5],x0)]
fx1 = [diff(xxp[0],x1), diff(xxp[1],x1), diff(xxp[2],x1), diff(xxp[3],x1), diff(xxp[4],x1), diff(xxp[5],x1)]
fx2 = [diff(xxp[0],x2), diff(xxp[1],x2), diff(xxp[2],x2), diff(xxp[3],x2), diff(xxp[4],x2), diff(xxp[5],x2)]
fx3 = [diff(xxp[0],x3), diff(xxp[1],x3), diff(xxp[2],x3), diff(xxp[3],x3), diff(xxp[4],x3), diff(xxp[5],x3)]
fx4 = [diff(xxp[0],x4), diff(xxp[1],x4), diff(xxp[2],x4), diff(xxp[3],x4), diff(xxp[4],x4), diff(xxp[5],x4)]
fx5 = [diff(xxp[0],x5), diff(xxp[1],x5), diff(xxp[2],x5), diff(xxp[3],x5), diff(xxp[4],x5), diff(xxp[5],x5)]

fu0 = [diff(xxp[0],u0), diff(xxp[1],u0), diff(xxp[2],u0), diff(xxp[3],u0), diff(xxp[4],u0), diff(xxp[5],u0)]
fu1 = [diff(xxp[0],u1), diff(xxp[1],u1), diff(xxp[2],u1), diff(xxp[3],u1), diff(xxp[4],u1), diff(xxp[5],u1)]


print (fx0,'\n\n', fx1,'\n\n', fx2,'\n\n', fx3,'\n\n', fx4,'\n\n', fx5,'\n\n', fu0,'\n\n', fu1)
