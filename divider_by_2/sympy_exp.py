from sympy import sympify, Matrix
from sympy import init_printing
import numpy as np

init_printing()


import json

with open("system.json", "r") as f:
    sys = json.load(f)

print(sys)

D = sympify(sys["mode"]["0"]["D"]["expression"]).subs(sys["mode"]["0"]["D"]["elements"])

#
# parameters = {"ron": 10, "cfly": 200e-12, "cload": 10e-9}
# variables = {"vin": 2, "iload": 5e-3}
#
# a00 = "-1 / cfly / ron / 2"
#
#
# a1_terms = {
#     "a11": "-1 / cfly / ron / 2",
#     "a12": "-1 / cfly / ron / 2",
#     "a21": "-1 / cload / ron / 2",
#     "a22": "-1 / cload / ron / 2",
# }
# a00s = sympify(a00)
# a00s.subs(parameters)
#
# A1 = Matrix([["a11", "a12"], ["a21", "a22"]])
