from jinja2 import Environment, FileSystemLoader
import os
from divider_by_2.matrices import *


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
    environment.filters["get_mode_indices"] = get_mode_indices
    template_out = environment.get_template(template_file).render(context)
    with open(os.path.join(output_dir, name), "w") as f:
        f.write(template_out)


def get_mode_indices(mode):
    return ", ".join(
        [
            ", ".join([f"a{n}{m}" for (n, m), x in np.ndenumerate(mode.A)]),
            ", ".join([f"b{n}{m}" for (n, m), x in np.ndenumerate(mode.B)]),
            ", ".join([f"c{n}{m}" for (n, m), x in np.ndenumerate(mode.C)]),
            ", ".join([f"d{n}{m}" for (n, m), x in np.ndenumerate(mode.D)]),
        ]
    )


template = "verilog_a_template"
context = {
    "name": "sw_cap_2_1",
    "parameters": {"ron": RON, "cfly": CFLY, "cload": CLOAD},
    "modes": [phase_1, phase_2],
    "nb_inputs": phase_1.inputs,
    "nb_outputs": phase_1.outputs,
    "nb_states": phase_1.states,
    "nb_modes": 2,
}

# a_indices = []
# for (n, m), x in np.ndenumerate(phase_1.A):
#    print("".join(f"a{n}{m}"))

a_indices = ", ".join(
    [
        ", ".join([f"a{n}{m}" for (n, m), x in np.ndenumerate(phase_1.A)]),
        ", ".join([f"b{n}{m}" for (n, m), x in np.ndenumerate(phase_1.B)]),
        ", ".join([f"c{n}{m}" for (n, m), x in np.ndenumerate(phase_1.C)]),
        ", ".join([f"d{n}{m}" for (n, m), x in np.ndenumerate(phase_1.D)]),
    ]
)

# print(get_mode_indices(phase_1))
# print(", ".join([f"a{n}{m}" for (n, m), x in np.ndenumerate(phase_1.A)]))
# print([",".join(n,m) for (n, m), x in np.ndenumerate(phase_1.A)])
render_template(template, context, ".", "sw_cap_2_1.vla")
