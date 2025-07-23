from generators.support.fpu import generate_fpu_wrapper


def generate_divf(name, params):
    core_unit = "Divider"
    mod_type = "divf"
    return generate_fpu_wrapper(name, params, core_unit, mod_type)