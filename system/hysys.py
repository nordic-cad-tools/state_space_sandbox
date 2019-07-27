from sympy import sympify, Matrix
import json
from json import JSONEncoder
from control import StateSpace
from control import forced_response
import matplotlib.pyplot as plt

import numpy as np


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


class Mode:
    def __init__(self, A, B, C, D):
        self.A = ModeMatrix(**A)
        self.B = ModeMatrix(**B)
        self.C = ModeMatrix(**C)
        self.D = ModeMatrix(**D)

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def to_json(self):
        return {
            "A": self.A.to_json(),
            "B": self.B.to_json(),
            "C": self.C.to_json(),
            "D": self.D.to_json(),
        }

    def to_ss(self, parameters):
        return StateSpace(
            np.array(self.A.eval(parameters)).astype(np.float64),
            np.array(self.B.eval(parameters)).astype(np.float64),
            np.array(self.C.eval(parameters)).astype(np.float64),
            np.array(self.D.eval(parameters)).astype(np.float64),
        )


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

    def transient(self, switching_instants, nb_periods, inputs=[], nb_points=10):

        # create the time vector for each phase
        t_vectors = [np.linspace(0, d, nb_points) for d in switching_instants]
        # create the input vector for each phase
        u_vectors = [np.array([np.ones_like(t) * i for i in inputs]) for t in t_vectors]
        # initialization of vectors
        t = np.zeros((1,))
        x = np.zeros((2, 1))
        y = np.zeros((1,))

        # hybrid system simulation
        for n in range(nb_periods):
            t_1, y_1, x_1 = forced_response(
                sys=self.modes[0].to_ss(self.parameters),
                T=t_vectors[0] + t[-1],
                U=u_vectors[0],
                X0=x[:, -1],
            )
            t = np.concatenate((t, t_1), axis=0)
            x = np.concatenate((x, x_1), axis=1)
            y = np.concatenate((y, y_1), axis=0)
            t_2, y_2, x_2 = forced_response(
                sys=self.modes[1].to_ss(self.parameters),
                T=t_vectors[1] + t[-1],
                U=u_vectors[1],
                X0=x[:, -1],
            )
            t = np.concatenate((t, t_2), axis=0)
            x = np.concatenate((x, x_2), axis=1)
            y = np.concatenate((y, y_2), axis=0)

        return t, x, y

if __name__ == "__main__":
    # load json file into sys
    with open("system.json", "r") as f:
        sys = json.load(f)
    # print(sys)
    # print(HySy.from_json(sys).to_json())
    my_sys = HySy.from_json(sys)
    # print(my_sys.modes[0].D.eval(my_sys.parameters))
    # print(my_sys.modes[0].to_ss(my_sys.parameters))

    t, x, y = my_sys.transient(switching_instants=[1e-8, 1e-8], nb_periods=50, inputs=[5e-3, 2])
    plt.figure()
    plt.plot(t, x[0], t, x[1])
    plt.plot(t, y)
    plt.show()

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
