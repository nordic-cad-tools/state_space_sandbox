import matplotlib.pyplot as plt
from divider_by_2.matrices import *
from control import forced_response


d1 = 1e-8
d2 = 1e-8
points = 10
nb_period = 200


t1 = np.linspace(0, d1, points)
t2 = np.linspace(0, d2, points)

u1 = np.array([np.ones_like(t1)*ILOAD, np.ones_like(t1)*VIN])
u2 = np.array([np.ones_like(t2) * ILOAD, np.ones_like(t2) * VIN])

t = np.array([0])
x = np.array([[0], [0]])
y = np.array([0])

for n in range(nb_period):
    t_1, y_1, x_1 = forced_response(sys=phase_1, T=t1+t[-1], U=u1, X0=x[:, -1])
    t = np.concatenate((t, t_1), axis=0)
    x = np.concatenate((x, x_1), axis=1)
    y = np.concatenate((y, y_1), axis=0)
    t_2, y_2, x_2 = forced_response(sys=phase_2, T=t2+t[-1], U=u2, X0=x_1[:, -1])
    t = np.concatenate((t, t_2), axis=0)
    x = np.concatenate((x, x_2), axis=1)
    y = np.concatenate((y, y_2), axis=0)

plt.figure()
plt.plot(t, x[0], t, x[1])
plt.show()