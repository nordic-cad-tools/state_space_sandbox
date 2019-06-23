import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response
from divider_by_2.fixed_point import fixed_point
from divider_by_2.find_steady_state import find_phase_duration
import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis, round
from scipy.linalg import expm, det, inv, logm, eig
from scipy import signal
import control

# vector to get the output voltage
E = np.array([0, 1])
# duration of each phase
# d1 = d2 = find_phase_duration([phase_1, phase_2], E=E, set_point=0.6)

Fs = 10e6
Ts = 1 / Fs
d1 = d2 = Ts / 2
# Ts = d1 + d2
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


# H = C(zI - A)-1 B +D
H = control.ss(phi_0, gam_0, E, D1, Ts)
# print(H)
# H = control.balred(H, 2, "truncate")
# print(Hr)


# convert to contrinuous using zoh method
na, ma = H.A.shape
nb, mb = H.B.shape
d_mat = np.concatenate(
    (
        np.concatenate((H.A, H.B), axis=1),
        np.concatenate((np.zeros_like(H.A), eye(nb, mb)), axis=1),
    ),
    axis=0,
)
c_mat = logm(d_mat) / Ts

phi_0_c = c_mat[0:na, 0:ma]
gam_0_c = c_mat[0:nb, ma:ma + mb]

# create the continuous state space
Hc = control.ss(phi_0_c, gam_0_c, E, D1)
# extract frequency response for the 2 inputs
w = np.logspace(3, 7)
mag_d, phase_d, wo_d = H.freqresp(w)
mag_c, phase_c, wo_c = Hc.freqresp(w)

mag_c_1, mag_c_2 = 20 * np.log10(np.squeeze(mag_c))
p_c_1, p_c_2 = np.rad2deg(np.squeeze(phase_c))

mag_d_1, mag_d_2 = 20 * np.log10(np.squeeze(mag_d))
p_d_1, p_d_2 = np.rad2deg(np.squeeze(phase_d))


plt.figure()
plt.subplot(211)
plt.semilogx(wo_c, mag_c_2)
plt.semilogx(wo_d, mag_d_2)
plt.grid(b=True, which="major", color="#666666", linestyle="-")
plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.subplot(212)
plt.semilogx(wo_c, p_c_2)
plt.semilogx(wo_d, p_d_2)
plt.grid(b=True, which="major", color="#666666", linestyle="-")
plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.figure()
plt.subplot(211)
plt.semilogx(wo_c, mag_c_1)
plt.semilogx(wo_d, mag_d_1)
plt.grid(b=True, which="major", color="#666666", linestyle="-")
plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.subplot(212)
plt.semilogx(wo_c, p_c_1)
plt.semilogx(wo_d, p_d_1)
plt.grid(b=True, which="major", color="#666666", linestyle="-")
plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.show()
