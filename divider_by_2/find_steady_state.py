from divider_by_2.matrices import *
from divider_by_2.fixed_point import fixed_point
from scipy.optimize import fsolve
from numpy import dot


def find_phase_duration(phases, E, set_point):
    phase_1, phase_2 = phases

    def fun(d, E, set_point):
        return dot(E, fixed_point([phase_1, phase_2], [d[0], d[0]])) - set_point

    d, info, ier, msg = fsolve(
        fun, 1e-8, full_output=True, args=(E, set_point), xtol=1e-12
    )
    if ier == 1:
        return d[0]
    else:
        raise RuntimeError
