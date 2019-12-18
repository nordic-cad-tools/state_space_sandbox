import json
from divider_by_2.matrices import *

# with open("system.json", 'r') as f:
#     sys = json.load(f)
#
# print(sys)


class HySys:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


sw = HySys()
sw.name = "sw_cap_2_1"
sw.parameters = {"ron": 10, "cfly": 200e-12, "cload": 10e-9}
sw.A1 = A1_11
print(sw.toJSON())
