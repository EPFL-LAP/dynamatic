from collections.abc import Callable
from .utils.types import ExtraSignals
from generators.support.signal_manager import generate_buffered_signal_manager

def generate_buffered_arith_signal_manager(
        name: str,
        bitwidth,
        extra_signals: ExtraSignals,
        generate_inner: Callable[[str], str],
        latency: int,
) -> str:
    return generate_buffered_signal_manager(
        name,
        [{
            "name": "lhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "rhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "result",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        generate_inner,
        latency
    )
