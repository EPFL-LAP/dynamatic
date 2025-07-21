from generators.support.fpu_wrapper import generate_fpu_wrapper

def generate_mulf(name, params):
    params["core_unit"] = "Multiplier"

    return generate_fpu_wrapper(name, params)