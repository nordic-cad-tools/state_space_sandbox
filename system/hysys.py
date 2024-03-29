from sympy import sympify, Matrix
import json
from json import JSONEncoder
from control import StateSpace
from control import forced_response
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot, eye, trapz, linspace, apply_along_axis, round
from scipy.linalg import expm, det, inv
from scipy.optimize import fsolve

from jinja2 import Environment, FileSystemLoader
import os


def render_template(template_file, context, output_dir, name):
    """
    Render a template with a context and save it into a file
    :param template_file: name of the template to be used
    :param context: context for rendering the template
    :param output_dir: directory where to save the output
    :param name: name of the output file
    """
    environment = Environment(
        autoescape=False, loader=FileSystemLoader("."), trim_blocks=False
    )
    template_out = environment.get_template(template_file).render(context)
    with open(os.path.join(output_dir, name), "w") as f:
        f.write(template_out)


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
        return [
            np.array(self.A.eval(parameters)).astype(np.float64),
            np.array(self.B.eval(parameters)).astype(np.float64),
            np.array(self.C.eval(parameters)).astype(np.float64),
            np.array(self.D.eval(parameters)).astype(np.float64),
        ]

    def to_ss(self, parameters):
        return StateSpace(*self.matrices(parameters))


class HySy:
    def __init__(self, name, parameters, modes):
        self.name = name
        self.parameters = parameters
        self.modes = modes
        self.nb_modes = len(self.modes)

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

    def to_verilog_a(self, output_dir="."):
        # context={
        #     "input_bus":
        # }

        environment = Environment(
            autoescape=False, loader=FileSystemLoader("."), trim_blocks=False
        )
        template_out = environment.get_template("verilog_a_template").render(jsonobj=self.to_json())
        with open(os.path.join(output_dir, f"{self.name}.vla"), "w") as f:
            f.write(template_out)


        # render_template(
        #     template_file="verilog_a_template",
        #     context=self.to_json(),
        #     output_dir=output_dir,
        #     name=f"{self.name}.vla",
        # )

    def transient(self, durations=None, nb_periods=10, inputs=[], nb_points=10):
        if durations is None:
            durations = [mode.duration for mode in self.modes]
        # create the time vector for each phase
        t_vectors = [np.linspace(0, d, nb_points) for d in durations]
        # create the input vector for each phase
        u_vectors = [np.array([np.ones_like(t) * i for i in inputs]) for t in t_vectors]
        # nb state in the system
        n, m = self.modes[0].A.to_array(self.parameters).shape
        # initialization of vectors
        t = np.zeros((1,))
        x = np.zeros((n, 1))
        y = np.zeros((1,))

        # hybrid system simulation
        for n in range(nb_periods):
            print(f"period {n}")
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

    def fixed_point(self, durations=None, inputs=[]):
        if durations is None:
            durations = [mode.duration for mode in self.modes]
        determinants = [
            round(det(mode.A.to_array(self.parameters))) for mode in self.modes
        ]

        # Identity matrix
        n, m = self.modes[0].A.to_array(self.parameters).shape
        I = eye(n, m)

        # phi calculation
        phi = 1
        phi_i = [
            expm(mode.A.to_array(self.parameters) * d)
            for mode, d in zip(self.modes, durations)
        ]
        for n, phi_n in enumerate(phi_i):
            phi = dot(phi_n, phi)

        # calculate the integral of exp matrix
        def expm_int(m, a, b, pt=100):
            def f(m, t):
                return expm(m * t)

            t = linspace(a, b, pt)
            result = apply_along_axis(f, 0, t.reshape(1, -1), m)
            return trapz(result, t)

        # gamma calculation
        # check for singularity
        if all(determinants):
            # non singular matrix
            gam_i = [
                dot(
                    dot((phi_n - I), inv(mode.A.to_array(self.parameters))),
                    mode.B.to_array(self.parameters),
                )
                for phi_n, mode in zip(phi_i, self.modes)
            ]
        else:
            # singular matrix
            gam_i = [
                dot(
                    expm_int(m=mode.A.to_array(self.parameters), a=0, b=d, pt=10000),
                    mode.B.to_array(self.parameters),
                )
                for phi_n, mode, d in zip(phi_i, self.modes, durations)
            ]

        gam = 0
        for k, gam_n in enumerate(gam_i):
            phi_tmp = 1
            for n, phi_n in enumerate(phi_i[:k:-1]):
                phi_tmp = dot(phi_n, phi_tmp)
            gam = dot(phi_tmp, gam_n) + gam

        # fixed point calculation
        X0 = dot(dot(inv(I - phi), gam), inputs)
        return X0

    def find_mode_durations(self, E, set_point, inputs):
        def fun(duration, E, set_point, inputs):
            durations = [duration[0]] * self.nb_modes
            x0 = self.fixed_point(durations=durations, inputs=inputs)
            return dot(E, x0 - set_point)[0]

        d, info, ier, msg = fsolve(
            fun, 1e-8, full_output=True, args=(E, set_point, inputs), xtol=1e-12
        )
        if ier == 1:
            return [d[0]] * self.nb_modes
        else:
            raise RuntimeError


if __name__ == "__main__":
    # load json file into sys
    with open("system.json", "r") as f:
        # with open("sw_cap_2_1_x4.json", "r") as f:

        sys = json.load(f)
    # print(sys)
    # print(HySy.from_json(sys).to_json())
    my_sys = HySy.from_json(sys)
    my_sys.to_verilog_a()
    # print(my_sys.modes[0].D.eval(my_sys.parameters))
    # print(my_sys.modes[0].to_ss(my_sys.parameters))

    # U = np.array([[5e-3], [2]])
    #
    # # d = my_sys.find_mode_durations(inputs=U, E=np.array([0, 1]), set_point=0.7)
    # # print(d)
    # #
    # x0 = my_sys.fixed_point(inputs=U)
    # print(x0)
    # #
    # t, x, y = my_sys.transient(durations=None, nb_periods=200, inputs=[5e-3, 2])
    # plt.figure()
    # for xi in x:
    #     plt.plot(t, xi)
    # plt.plot(t, y)
    # for x0i in x0:
    #     plt.plot(t[-1], x0i, "o")
    # # plt.plot(t[-1], x0[0], "o")
    # # plt.plot(t[-1], x0[1], "o")
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
