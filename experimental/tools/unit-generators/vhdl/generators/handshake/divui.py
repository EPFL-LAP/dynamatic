from generators.support.fpu import generate_fpu_wrapper


def generate_divui(name, params):
    fpu_impl = params["fpu_impl"]

    if fpu_impl != "vivado":
        raise ValueError(f"Invalid divui implementation: {fpu_impl}")

    mod_type = "divui"

    # only applies to flopoco
    core_unit = ""

    return generate_fpu_wrapper(name, params, core_unit, mod_type)
