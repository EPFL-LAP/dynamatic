from generators.support.fpu_wrapper import generate_fpu_wrapper

def generate_divf(name, params):
    params["core_unit"] = "Divider"

    return generate_fpu_wrapper(name, params)