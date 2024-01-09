
import numpy as np

T = 10
H = np.zeros((T,1))

for i in range(T-1):

    H [i] = i

print(max(H[0,:]))