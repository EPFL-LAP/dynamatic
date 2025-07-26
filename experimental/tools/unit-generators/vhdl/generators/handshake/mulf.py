from generators.support.arith_ip import generate_flopoco_ip_wrapper, generate_vivado_ip_wrapper


def generate_mulf(name, params):
    impl = params["fpu_impl"]
    latency = params["latency"]

    extra_signals = params.get("extra_signals", None)

    # only used for flopoco
    is_double = params.get("is_double", None)
    internal_delay = params.get("internal_delay", None)


    mod_type = "mulf"

    if impl == "flopoco":
        if is_double is None:
            raise ValueError(f"is_double was missing for generating a flopoco {mod_type}")
        if internal_delay is None:
            raise ValueError(f"internal_delay was missing for generating a flopoco {mod_type}")

        return generate_flopoco_ip_wrapper(
            name=name,
            mod_type=mod_type,
            core_unit="FloatingPointMultiplier",
            latency=latency,
            is_double=is_double,
            internal_delay=internal_delay,
            extra_signals=extra_signals
        )
    elif impl == "vivado":
        return generate_vivado_ip_wrapper(
            name=name,
            mod_type=mod_type,
            latency=latency,
            extra_signals=extra_signals
        )
