from generators.support.arith_ip import generate_vivado_ip_wrapper


def generate_divui(name, params):

    latency = params["latency"]

    extra_signals = params.get("extra_signals", None)

    return generate_vivado_ip_wrapper(
        name=name,
        op_type="divui",
        latency=latency,
        extra_signals=extra_signals
    )
