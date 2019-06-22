import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response
from divider_by_2.fixed_point import fixed_point
from divider_by_2.find_steady_state import find_phase_duration
import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis, round
from scipy.linalg import expm, det, inv


# duration of each phase
d1 = d2 = find_phase_duration([phase_1, phase_2], E=np.array([0, 1]), set_point=0.7)

# if Ai are not singular
# used scipy instead of numpy, it gives ad results. Also need to use np.array and not matrix
det_A1 = round(det(phase_1.A))
det_A2 = round(det(phase_2.A))

# Identity matrix
n, m = A1.shape
I = eye(n, m)

# duration of each phase
# d1 = 1e-8
# d2 = 1e-8

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
# phase 1 fixed point
X0d = dot(phi_1, X0) + dot(gam_1, U)

# linearized matrices
phi_0 = dot(phi_2, phi_1)
gam_0 = dot(phi_2, gam_1) + gam_2
gam_d = dot(dot(phi_2, (phase_1.A - phase_2.A)), X0d) + dot(phase_1.B - phase_2.B, U)
gam_t = dot(phi_2, dot(phase_2.A, X0d) + dot(phase_2.B, U))

