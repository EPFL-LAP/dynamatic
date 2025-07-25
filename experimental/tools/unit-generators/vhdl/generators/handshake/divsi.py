from generators.support.fpu import generate_fpu_wrapper


def generate_divsi(name, params):
    params["fpu_impl"] = "vivado"

    mod_type = "divsi"

    # only applies to flopoco
    core_unit = ""

    return generate_fpu_wrapper(name, params, core_unit, mod_type)
