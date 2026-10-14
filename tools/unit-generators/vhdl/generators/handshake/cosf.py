from generators.support.arith_ip import generate_vitis_hls_unary_ip_wrapper


def generate_cosf(name, params):
    bitwidth = params["bitwidth"]
    if bitwidth != 64:
        raise ValueError(f"cosf only supports 64-bit operands, got {bitwidth}")

    return generate_vitis_hls_unary_ip_wrapper(
        name=name,
        handshake_op="cosf",
        bitwidth=bitwidth,
        latency=params["latency"],
        ip_latency=33,
        extra_signals=params.get("extra_signals", None)
    )
