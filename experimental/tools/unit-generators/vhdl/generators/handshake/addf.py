from generators.support.fpu import generate_fpu_wrapper


def generate_addf(name, params):
    core_unit = "FloatingPointAdder"
    mod_type = "addf"
    return generate_fpu_wrapper(name, params, core_unit, mod_type)
