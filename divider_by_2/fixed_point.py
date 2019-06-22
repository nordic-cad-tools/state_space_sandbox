from divider_by_2.matrices import *
import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis
from scipy.linalg import expm, det, inv


# if Ai are not singular
# used scipy instead of numpy, it gives ad results. Also need to use np.array and not matrix
det_A1 = det(np.array(A1))
det_A2 = det(phase_2.A)
# Identity matrix
n, m = phase_1.A.shape
I = eye(n, m)

# duration of each phase
d1 = 1e-8
d2 = 1e-8

# phi calculation
phi_1 = expm(phase_1.A * d1)
phi_2 = expm(phase_2.A * d2)
phi = dot(phi_2, phi_1)

# calculate the integral of exp matrix
def expm_int(A, T, pt=100):
    def f(A, t):
        return expm(A * t)

    t = linspace(0, T, pt)
    result = apply_along_axis(f, 0, t.reshape(1, -1), A)
    return trapz(result, t)


# gamma calculation
# check for singularity
if all([det_A1, det_A2]):
    # non singular matrix
    gam_1 = dot(dot((phi_1 - I), inv(phase_1.A)), phase_1.B)
    gam_2 = dot(dot((phi_2 - I), inv(phase_2.A)), phase_2.B)
else:
    # singular matrix
    gam_1 = dot(expm_int(A1, d1), B1)
    gam_2 = dot(expm_int(A2, d2), B2)

gam = dot(phi_2, gam_1) + gam_2

# fixed point calculation
X0 = dot(dot(inv(I - phi), gam), U)
