import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response
from divider_by_2.fixed_point import fixed_point
from divider_by_2.find_steady_state import find_phase_duration
import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis, round
from scipy.linalg import expm, det, inv
from scipy import signal
import control

# vector to get the output voltage
E = np.array([0, 1])
# duration of each phase
d1 = d2 = find_phase_duration([phase_1, phase_2], E=E, set_point=0.7)

Ts = d1 + d2
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

# transfer function
num = [1, 3, 3]
den = [1, 2, 1]

# H = C(zI - A)-1 B +D
H = control.ss(phi_0, gam_0, E, D1, Ts)
print(H)
Hr = control.balred(H, 1, "truncate")
print(Hr)

mag, phase, w = control.bode(H)
# plt.figure()
# plt.semilogx(w, mag)    # Bode magnitude plot
# plt.figure()
# plt.semilogx(w, phase)  # Bode phase plot
# plt.grid(True)
# plt.show()
# tzu = dot(E, inv())
# output_args.Tzu=(HySimSys.E/(z*I-PHI0)*GAM0);
# output_args.Tzd=(HySimSys.E/(z*I-PHI0)*GAMd);
# output_args.Tzt=(HySimSys.E/(z*I-PHI0)*GAMt);
# output_args.Tzf=-1*output_args.Tzt*OperatingPoint.T*OperatingPoint.T;
#
# method='Truncate';
# output_args.rTzu=balred(output_args.Tzu, order, 'Elimination',method);
# output_args.rTzd=balred(output_args.Tzd, order, 'Elimination',method);
# output_args.rTzT=balred(output_args.Tzt, order, 'Elimination',method);
# output_args.rTzF=balred(output_args.Tzf, order, 'Elimination',method);
# output_args.Tcu=d2c(output_args.rTzu,z2s);
# output_args.Tcd=d2c(output_args.rTzd,z2s);
# output_args.TcT=d2c(output_args.rTzT,z2s);
# output_args.TcF=d2c(output_args.rTzF,z2s);
