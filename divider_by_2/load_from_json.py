from sympy import init_printing
from sympy import sympify, Matrix
from control import StateSpace
import numpy as np
import json


init_printing()

# load json file into sys
with open("system.json", "r") as f:
    sys = json.load(f)

# Creating state space
# 1. Get matrix expression
# 2. Substitute with element expressions
# 3. Substitute with the parameters
# 4. Pack as a list

phases = []
for num, mat in sys["mode"].items():
    phases.append(
        StateSpace(
            np.array(
                sympify(mat["A"]["expression"])
                .subs(mat["A"]["elements"])
                .subs(sys["parameters"])
            ).astype(np.float64),
            np.array(
                sympify(mat["B"]["expression"])
                .subs(mat["B"]["elements"])
                .subs(sys["parameters"])
            ).astype(np.float64),
            np.array(
                sympify(mat["C"]["expression"])
                .subs(mat["C"]["elements"])
                .subs(sys["parameters"])
            ).astype(np.float64),
            np.array(
                sympify(mat["D"]["expression"])
                .subs(mat["D"]["elements"])
                .subs(sys["parameters"])
            ).astype(np.float64),
        )
    )

phase_1, phase_2 = phases
