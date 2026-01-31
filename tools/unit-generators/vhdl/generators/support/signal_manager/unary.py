from collections.abc import Callable
from .utils.types import ExtraSignals
from generators.support.signal_manager import generate_buffered_signal_manager, generate_default_signal_manager


def generate_unary_signal_manager(
        name: str,
        extra_signals: ExtraSignals,
        generate_inner: Callable[[str], str],
        latency: int = 0,
        bitwidth: int = None,
        input_bitwidth: int = None,
        output_bitwidth: int = None,
) -> str:
    if bitwidth is not None:
        if input_bitwidth is not None or output_bitwidth is not None:
            raise RuntimeError("If bitwidth is specified, input and output bitwidth must not be specified")

        input_bitwidth = bitwidth
        output_bitwidth = bitwidth

    elif input_bitwidth is None or output_bitwidth is None:
        raise RuntimeError("If bitwidth is not specified, both input and output bitwidth must be specified")

    in_channels = \
        [{
            "name": "ins",
            "bitwidth": input_bitwidth,
            "extra_signals": extra_signals
        }]
    out_channels = \
        [{
            "name": "outs",
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
