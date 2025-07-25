from generators.support.fpu import generate_fpu_wrapper


def generate_mulf(name, params):
    core_unit = "FloatingPointMultiplier"
    mod_type = "mul"
    return generate_fpu_wrapper(name, params, core_unit, mod_type)
