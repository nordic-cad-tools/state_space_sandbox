import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response
import numpy as np
from numpy import dot
#from numpy.linalg import inv, slogdet
from scipy.linalg import expm, det, inv

# if Ai are not singular
det_A1 = det(np.array(A1))
det_A2 = det(phase_2.A)
# Identity matrix
n, m = phase_1.A.shape
I = np.eye(n, m)

# duration of each phase
d1 = 1e-8
d2 = 1e-8

#
# phi_1 = expm(phase_1.A * d1)
# phi_2 = expm(phase_2.A * d2)
# phi = dot(phi_2, phi_1)
#
# gam_1 = dot(dot((phi_1 - I), inv(phase_1.A)), phase_1.B)
# gam_2 = dot(dot((phi_2 - I), inv(phase_2.A)), phase_2.B)
# gam = dot(phi_2, gam_1) + gam_2
# X0 = dot(dot(inv(I - phi), gam), U)
