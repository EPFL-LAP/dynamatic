from generators.support.fpu import generate_fpu_wrapper


def generate_divsi(name, params):
    fpu_impl = params["fpu_impl"]

    if fpu_impl != "vivado":
        raise ValueError(f"Invalid divsi implementation: {fpu_impl}")

    mod_type = "divsi"

    # only applies to flopoco
    core_unit = ""

    return generate_fpu_wrapper(name, params, core_unit, mod_type)
