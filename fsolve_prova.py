
import numpy as np

TT = 10
d1l=np.zeros((10,1))
d2l=[1,2,3,4,5,6,7,8,9,10]

for tt in reversed(range(TT)):                        # integration backward in time

    d1l[tt] = d2l[tt] 

print (d1l[0])