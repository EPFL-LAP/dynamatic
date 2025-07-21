from generators.support.fpu_wrapper import generate_fpu_wrapper

def generate_addf(name, params):
    params["core_unit"] = "Adder"

    return generate_fpu_wrapper(name, params)