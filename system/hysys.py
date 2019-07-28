from sympy import sympify, Matrix
import json
from json import JSONEncoder
from control import StateSpace
from control import forced_response
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis, round
from scipy.linalg import expm, det, inv

class ModeMatrix:
    def __init__(self, elements, expression):
        """
        Mode matrix representation.
        :param elements: A dict mapping all elements of the matrix
        :param expression: A sympy expression string reprenting the matrix
        """
        self.elements = elements
        self.expression = sympify(expression)

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def to_json(self):
        return {"elements": self.elements, "expression": self.expression.__str__()}

    @property
    def symbol(self):
        """
        This function returns the sympy matrix using the elements expression
        :return: sympy matric
        """
        return self.expression.subs(self.elements)

    def eval(self, parameters):
        """
        Evaluate the sympy matrix according to parameters
        :param parameters: A dict of parameters
        :return: sympy matrix
        """
        return self.symbol.subs(parameters)

    def to_array(self, parameters):
        return np.array(self.eval(parameters)).astype(np.float64)

class Mode:
    def __init__(self, A, B, C, D, duration):
        self.A = ModeMatrix(**A)
        self.B = ModeMatrix(**B)
        self.C = ModeMatrix(**C)
        self.D = ModeMatrix(**D)
        self.duration = duration

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def to_json(self):
        return {
            "duration": self.duration,
            "A": self.A.to_json(),
            "B": self.B.to_json(),
            "C": self.C.to_json(),
            "D": self.D.to_json(),
        }

    def matrices(self, parameters):
        return [np.array(self.A.eval(parameters)).astype(np.float64),
            np.array(self.B.eval(parameters)).astype(np.float64),
            np.array(self.C.eval(parameters)).astype(np.float64),
            np.array(self.D.eval(parameters)).astype(np.float64)]

    def to_ss(self, parameters):
        return StateSpace(*self.matrices(parameters))


class HySy:
    def __init__(self, name, parameters, modes):
        self.name = name
        self.parameters = parameters
        self.modes = modes

    @classmethod
    def from_json(cls, data):
        obj = cls(
            name=data["name"],
            parameters=data["parameters"],
            modes=[Mode.from_json(mode) for mode in data["modes"]],
        )
        return obj

    def to_json(self):
        return {
            "name": self.name,
            "parameters": self.parameters,
            "modes": [mode.to_json() for mode in self.modes],
        }

    def transient(self, durations, nb_periods, inputs=[], nb_points=10):

        # create the time vector for each phase
        t_vectors = [np.linspace(0, d, nb_points) for d in durations]
        # create the input vector for each phase
        u_vectors = [np.array([np.ones_like(t) * i for i in inputs]) for t in t_vectors]
        # initialization of vectors
        t = np.zeros((1,))
        x = np.zeros((2, 1))
        y = np.zeros((1,))

        # hybrid system simulation
        for n in range(nb_periods):

            for mode, t_vector, u_vector in zip(self.modes, t_vectors, u_vectors):
                t_i, y_i, x_i = forced_response(
                    sys=mode.to_ss(self.parameters),
                    T=t_vector + t[-1],
                    U=u_vector,
                    X0=x[:, -1],
                )
                t = np.concatenate((t, t_i), axis=0)
                x = np.concatenate((x, x_i), axis=1)
                y = np.concatenate((y, y_i), axis=0)

        return t, x, y

    def fixed_point(self, durations, inputs=[]):
        matrices = [mode.matrices(self.parameters) for mode in self.modes]

        A1, B1, C1 ,D1 = self.modes[0].matrices(self.parameters)
        A2, B2, C2 ,D2 = self.modes[1].matrices(self.parameters)

        # phi calculation
        phi = 1
        for mode in self.modes:
            phi = dot(expm(mode.A.to_array(self.parameters) * mode.duration), phi)

        print(phi)
        # tempory soution for 2 phases system
        d1, d2 = durations
        # if Ai are not singular
        # used scipy instead of numpy, it gives ad results. Also need to use np.array and not matrix
        det_A1 = round(det(A1))
        det_A2 = round(det(A2))

        # Identity matrix
        n, m = A1.shape
        I = eye(n, m)

        # duration of each phase
        # d1 = 1e-8
        # d2 = 1e-8

        # phi calculation
        phi_1 = expm(A1 * d1)
        phi_2 = expm(A2 * d2)
        phi = dot(phi_2, phi_1)
        print(phi)
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
            gam_1 = dot(dot((phi_1 - I), inv(A1)), B1)
            gam_2 = dot(dot((phi_2 - I), inv(A2)), B2)
        else:
            # singular matrix
            gam_1 = dot(expm_int(A1, d1), B1)
            gam_2 = dot(expm_int(A2, d2), B2)

        gam = dot(phi_2, gam_1) + gam_2

        # fixed point calculation
        X0 = dot(dot(inv(I - phi), gam), inputs)
        return X0


if __name__ == "__main__":
    # load json file into sys
    with open("system.json", "r") as f:
        sys = json.load(f)
    # print(sys)
    # print(HySy.from_json(sys).to_json())
    my_sys = HySy.from_json(sys)
    # print(my_sys.modes[0].D.eval(my_sys.parameters))
    # print(my_sys.modes[0].to_ss(my_sys.parameters))

    U = np.array([[5e-3], [2]])

    x0 = my_sys.fixed_point([1e-8, 1e-8], inputs=U)
    # print(x0)
    # t, x, y = my_sys.transient(durations=[1e-8, 1e-8], nb_periods=50, inputs=[5e-3, 2])
    # plt.figure()
    # plt.plot(t, x[0], t, x[1])
    # plt.plot(t, y)
    # plt.plot(t[-1], x0[0], "o")
    # plt.plot(t[-1], x0[1], "o")
    # plt.show()
    # MODE = {
    #     "A": {
    #         "elements": {
    #             "a11": "-1 / cfly / ron / 2",
    #             "a12": "-1 / cfly / ron / 2",
    #             "a21": "-1 / cload / ron / 2",
    #             "a22": "-1 / cload / ron / 2",
    #         },
    #         "expression": "Matrix([[a11, a12], [a21, a22]])",
    #     },
    #     "B": {
    #         "elements": {
    #             "b11": "0",
    #             "b12": "1 / cfly / ron / 2",
    #             "b21": "-1 / cload",
    #             "b22": "1 / cload / ron / 2",
    #         },
    #         "expression": "Matrix([[b11, b12], [b21, b22]])",
    #     },
    #     "C": {
    #         "elements": {"c11": "0", "c12": "1"},
    #         "expression": "Matrix([[c11, c12]])",
    #     },
    #     "D": {
    #         "elements": {"d11": "0", "d12": "0"},
    #         "expression": "Matrix([[d11, d12]])",
    #     },
    # }
    #
    # A = {
    #         "elements": {
    #             "a11": "-1 / cfly / ron / 2",
    #             "a12": "-1 / cfly / ron / 2",
    #             "a21": "-1 / cload / ron / 2",
    #             "a22": "-1 / cload / ron / 2",
    #         },
    #         "expression": "Matrix([[a11, a12], [a21, a22]])",
    #     }
    #
    # mode = Mode.from_json(MODE)
    # print(mode.to_json())

#
#     a = ModeMatrix.from_json(A)
#     print(type(a))
#     print(a.__dict__)
#
#
#     print(a.expression)
#     print(a.eval(parameters={"ron": 10, "cfly": 200e-12, "cload": 10e-9}))
#
# #    data = json.dumps(a, cls=ModeMatrix)
#     print(a.to_json())
#     print(mode.to_json())


# sys = HySy(
#     name="sw_cap_2_1", parameters={"ron": 10, "cfly": 200e-12, "cload": 10e-9}
# )
#
# print(
#     HySy.from_json(
#         {
#             "name": "sw_cap_2_1",
#             "parameters": {"ron": 10, "cfly": 200e-12, "cload": 10e-9},
#         }
#     ).__dict__
# )
