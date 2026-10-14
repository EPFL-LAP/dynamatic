from generators.support.arith_ip import generate_vitis_hls_unary_ip_wrapper


def generate_sqrtf(name, params):
    bitwidth = params["bitwidth"]
    if bitwidth != 32:
        raise ValueError(f"sqrtf only supports 32-bit operands, got {bitwidth}")

    return generate_vitis_hls_unary_ip_wrapper(
        name=name,
        handshake_op="sqrtf",
        bitwidth=bitwidth,
        latency=params["latency"],
        ip_latency=11,
        extra_signals=params.get("extra_signals", None)
    )
