from collections.abc import Callable
from .utils.types import ExtraSignals
from generators.support.signal_manager import generate_buffered_signal_manager, generate_default_signal_manager


def generate_arith_binary_signal_manager(
        name: str,
        input_bitwidth,
        output_bitwidth,
        extra_signals: ExtraSignals,
        generate_inner: Callable[[str], str],
        latency: int,
) -> str:
    in_channels = \
        [{
            "name": "lhs",
            "bitwidth": input_bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "rhs",
            "bitwidth": input_bitwidth,
            "extra_signals": extra_signals
        }]
    out_channels = \
        [{
            "name": "result",
            "bitwidth": output_bitwidth,
            "extra_signals": extra_signals
        }]

    if latency == 0:
        return generate_default_signal_manager(
            name,
            in_channels,
            out_channels,
            extra_signals,
            generate_inner
        )
    else:
        return generate_buffered_signal_manager(
            name,
            in_channels,
            out_channels,
            extra_signals,
            generate_inner,
            latency
        )
