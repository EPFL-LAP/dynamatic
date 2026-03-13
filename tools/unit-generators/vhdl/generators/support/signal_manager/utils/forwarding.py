# Default expression to use when no input channels are present
def get_default_extra_signal_value(extra_signal_name: str):
    """
    Return the default VHDL value for an extra signal
    when there are no input sources to forward from.
    """
    return "\"0\""


def generate_forwarding_expression_for_signal(signal_name: str, in_extra_signal_names: list[str]) -> str:
    """
    Generate a VHDL expression to forward an extra signal
    based on a list of input extra signal names.

    If the list is empty, a default value is returned.
    Currently, only the "spec" signal is supported,
    which is forwarded using a logical OR.

    Example: "0", lhs_spec or rhs_spec
    """

    if not in_extra_signal_names:
        return get_default_extra_signal_value(signal_name)

    if signal_name == "spec":
        return " or ".join(in_extra_signal_names)

    """
    Tags are guaranteed to be the same across all input ports.
    We can use the first input port's tag for all output ports.
    """
    if signal_name.startswith("tag"):
        if in_extra_signals:
            return in_extra_signals[0]
        raise ValueError(f"{signal_name} requires at least one signal")

    raise ValueError(
        f"Unsupported forwarding method for extra signal: {signal_name}")
