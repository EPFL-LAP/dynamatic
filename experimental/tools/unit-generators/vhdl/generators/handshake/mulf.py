from generators.support.arith_ip import generate_flopoco_ip_wrapper, generate_vivado_ip_wrapper
from generators.support.utils import VIVADO_IMPL, FLOPOCO_IMPL


def generate_mulf(name, params):
    impl = params["fpu_impl"]
    latency = params["latency"]

    extra_signals = params.get("extra_signals", None)

    # only used for flopoco
    is_double = params.get("is_double", None)
    internal_delay = params.get("internal_delay", None)

    op_type = "mulf"

    if impl == FLOPOCO_IMPL:
        if is_double is None:
            raise ValueError(f"is_double was missing for generating a flopoco {op_type}")
        if internal_delay is None:
            raise ValueError(f"internal_delay was missing for generating a flopoco {op_type}")

        return generate_flopoco_ip_wrapper(
            name=name,
            op_type=op_type,
            core_unit="FloatingPointMultiplier",
            latency=latency,
            is_double=is_double,
            internal_delay=internal_delay,
            extra_signals=extra_signals
        )
    elif impl == VIVADO_IMPL:
        return generate_vivado_ip_wrapper(
            name=name,
            op_type=op_type,
            latency=latency,
            extra_signals=extra_signals
        )
