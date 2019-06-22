import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response
import numpy as np
from numpy import dot

# from numpy.linalg import inv, slogdet
from scipy.linalg import expm, det, inv
from scipy.integrate import odeint
from scipy import integrate
from sympy import Matrix

# if Ai are not singular
# used scipy instead of numpy, it gives ad results. Also need to use np.array and not matrix
det_A1 = det(np.array(A1))
det_A2 = det(phase_2.A)
# Identity matrix
n, m = phase_1.A.shape
I = np.eye(n, m)

# duration of each phase
d1 = 1e-8
d2 = 1e-8


phi_1 = expm(phase_1.A * d1)
phi_2 = expm(phase_2.A * d2)
phi = dot(phi_2, phi_1)

print(phi_1)
print(phi_2)
print(phi)

print("o=> integration")
texpA1 = lambda t: expm(A1 * t)
x = np.linspace(0, d1, 1000)
y = map(texpA1, x)
# print(list(y))
# print(integrate.simps(texpA1))
#
def expm_int(A, T, pt=100):
    def f(A, t):
        return expm(A * t)

    t = np.linspace(0, T, pt)
    result = np.apply_along_axis(f, 0, t.reshape(1, -1), A)
    return np.trapz(result, t)

gam_1 = dot(expm_int(A1, d1), B1)
gam_2 = dot(expm_int(A2, d2), B2)

print(gam_1)
print(gam_2)
#GAM2=integral(expA2,0,Tk-dk,'ArrayValued',1)*HySimSys.HB(:,:,2);

#print(expm_int(A1, d1))
# print(integrate.trapz(texpA1(x)))
# print(odeint(texpA1, y0=0, t=np.linspace(0, d1, 1000)))
# GAM1=integral(expA1,0,dk,'ArrayValued',1)*HySimSys.HB(:,:,1);
#P, J = Matrix(A1).jordan_form()
# print(P)
# print(J)
# print(P.inv()*A1*P)
# Matrix.integrate()
# print(dot(P,expm(J)))

#
# gam_1 = dot(dot((phi_1 - I), inv(phase_1.A)), phase_1.B)
# gam_2 = dot(dot((phi_2 - I), inv(phase_2.A)), phase_2.B)
gam = dot(phi_2, gam_1) + gam_2
X0 = dot(dot(inv(I - phi), gam), U)
print(gam)
print(X0)
