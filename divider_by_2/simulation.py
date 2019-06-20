# # %% Imports
# # import numpy as np
import matplotlib.pyplot as plt
# # from scipy.integrate import solve_ivp
# # #from ode_helpers import state_plotter
# #
# # # %% Define derivative function
# # def f(t, y, c):
# #     dydt = [c[0]*np.cos(c[1]*t), c[2]*y[0]+c[3]*t]
# #     return dydt
# #
# # def dx()
# # # %% Define time spans, initial values, and constants
# # tspan = np.linspace(0, 5, 100)
# # yinit = [0, -3]
# # c = [4, 3, -2, 0.5]
# #
# # # %% Solve differential equation
# # sol = solve_ivp(lambda t, y: f(t, y, c),
# #                 [tspan[0], tspan[-1]], yinit, t_eval=tspan)
# #
# #
# # plt.plot(sol.t, sol.y[0])
# # plt.show()
# # # %% Plot states
# # #state_plotter(sol.t, sol.y, 1)


from scipy.signal import StateSpace
from scipy.signal import lsim, step
from divider_by_2.matrices import *
from scipy.integrate import odeint, RK23, solve_ivp
from control import matlab

import control
#print(A1)
#print(B1)
# def dx(x, t):
#     my_x = np.array([[x[0]], [x[1]]])
#     dx = np.dot(A1, my_x) + np.dot(B1, U)
#     return dx
#
# tspan = np.linspace(0, 15, 1000)
# x0 = np.zeros((2, 1))
# #print(np.zeros((2, 1)))
# yinit = np.array([0, 0])
# #print(x0)
# print(dx([0,0],0))

t = np.linspace(0,1e-7,100)
#y = RK23(fun=dx, t0=0,y0=x0, t_bound=1e-3)
#xx=odeint(dx, [0, 0], t)
#sol = solve_ivp(dx, [tspan[0], tspan[-1]], [0, 0])

#sys_1 = StateSpace(A1, B1, C1, D1)
#t,y =step(sys_1)
#
# t = np.linspace(0,5,5).reshape(1,5)
# u = np.zeros((2,5))
# #print(t.size, u.size)
# print(t)
# print(u)
#
# tout, y, x = lsim(sys_1, u, t)
# #print(u.reshape(50,2))
# #step2(sys_1)

A = [[-2.0, -6.0], [-8.0, -8.0]]
B = [[8.0], [0.0]]
C = [[0.0, 1.0], [1.0, 0.0]]
D = [[0.0], [0.0]]
sys = StateSpace(A, B, C, D)
t4, y4 = step(sys)


sys_1 = control.StateSpace(A1, B1, C1, D1)
sys_2 = control.StateSpace(A2, B2, C2, D2)
d1 = 1e-8
d2 = 1e-8
Nb = 1000
N_period = 200


t1 = np.linspace(0, d1, Nb)
t2 = np.linspace(0, d2, Nb)
# t3 = np.linspace(d2, d1+d2, Nb)
# t4 = np.linspace(d1+d2, d1+d2+d2, Nb)

u1 = np.array([np.ones_like(t1)*ILOAD, np.ones_like(t1)*VIN])
u2 = np.array([np.ones_like(t2) * ILOAD, np.ones_like(t2) * VIN])

# print(t1)
# print(t2)
# print(t3)
# print(t4)
#print(u1.tolist())
#T_1, y_1, x_1 = control.forced_response(sys=sys_1, T=t1, U=u1, X0=[[0], [0]])
#print()
#T, yout, xout = control.forced_response(sys, T, u, X0)
# yout, T, xout = matlab.lsim(sys, )
# T_2, y_2, x_2 = control.forced_response(sys=sys_2, T=t2, U=u2, X0=)

# plt.plot(T_1, x_1[0])
# plt.plot(T_2, x_2[0])

#plt.show()
# x = np.array([])
# t = np.array([])
# y = np.array([])
# t_1, y_1, x_1 = control.forced_response(sys=sys_1, T=t1, U=u1, X0=[[0], [0]])
# #print(x_1)
#
# #print(x_1[:,-1])
# # print(x)
# # t_2, y_2, x_2 = control.forced_response(sys=sys_2, T=t2, U=u2, X0=[[x[0][-1]], [x[1][-1]]])
# t_2, y_2, x_2 = control.forced_response(sys=sys_2, T=t2, U=u2, X0=x_1[:,-1])
#
# t = np.concatenate((t_1, t_2), axis=0)
# x = np.concatenate((x_1, x_2), axis=1)

X0 = [[0], [0]]
for n in range(N_period):
    print(f"Period {n}")
    if n == 0:
        t_1, y_1, x_1 = control.forced_response(sys=sys_1, T=t1, U=u1, X0=X0)
        t_2, y_2, x_2 = control.forced_response(sys=sys_2, T=t_1[-1] + t2, U=u2, X0=x_1[:, -1])
        t = np.concatenate((t_1, t_2), axis=0)
        x = np.concatenate((x_1, x_2), axis=1)
        y = np.concatenate((y_1, y_2), axis=0)

    else:
        t_1, y_1, x_1 = control.forced_response(sys=sys_1, T=t1+t[-1], U=u1, X0=x[:, -1])
        t = np.concatenate((t, t_1), axis=0)
        x = np.concatenate((x, x_1), axis=1)
        y = np.concatenate((y, y_1), axis=0)
        t_2, y_2, x_2 = control.forced_response(sys=sys_2, T=t2+t[-1], U=u2, X0=x_1[:, -1])
        t = np.concatenate((t, t_2), axis=0)
        x = np.concatenate((x, x_2), axis=1)
        y = np.concatenate((y, y_2), axis=0)

# print(x.shape)
# print(x)
# t.extend(t_2)
# y.extend(y_2)
# x.extend(x_2)
#plt.plot(t_1, x_1[0])
#plt.plot(t_2, x_2[0])
plt.figure()
plt.plot(t, x[0], t, x[1])
plt.show()
# for n in range(N_period):
#     print(n)
#     t_1, y_1, x_1 = control.forced_response(sys=sys_1, T=t1, U=u1, X0=[[x[0][-1]], [x[1][-1]]])
#     t_2, y_2, x_2 = control.forced_response(sys=sys_1, T=t1, U=u1, X0=[[x[0][-1]], [x[1][-1]]])

