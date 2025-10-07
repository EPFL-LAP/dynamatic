from .types import Channel, ExtraSignals


def create_internal_signal_decl(name: str) -> str:
    return f"wire {name};"


def create_internal_vector_decl(name: str, bitwidth: int) -> str:
    return f"wire [{bitwidth - 1} : 0]{name};"


def create_internal_array_decl(name: str, bitwidth: int, size: int) -> str:
    return f"wire [{bitwidth - 1} : 0] {name} [{size - 1} : 0];"


def create_internal_extra_signals_decl(channel_name: str, extra_signals: ExtraSignals) -> list[str]:
    """
    Generate VHDL declarations for extra signals in a channel.
    Example: When `extra_signals` is: {"spec": 1, "tag0": 8}
      wire [0:0] my_channel_spec;
      wire [7:0] my_channel_tag0;
    """

    return [
        create_internal_vector_decl(
            f"{channel_name}_{signal_name}", signal_bitwidth)
        for signal_name, signal_bitwidth in extra_signals.items()
    ]


def create_internal_channel_decl(channel: Channel) -> list[str]:
    """
    Generate VHDL declarations for a channel.
    Args:
      channel (Channel): Channel to generate declarations for.
    Returns:
      decls (list[str]): List of VHDL declarations for the channel.
      - Example:
        wire [31:0] my_channel;
        wire my_channel_valid;
        wire my_channel_ready;
        wire [0:0] my_channel_spec;
        wire [7:0] my_channel_tag0;
    """

    decls = []

    size = channel.get("size", 0)
    if size == 0:
        # Declare data signal if present
        if channel["bitwidth"] > 0:
            decls.append(create_internal_vector_decl(
                channel["name"], channel["bitwidth"]))

        # Declare handshake signals
        decls.append(create_internal_signal_decl(
            f"{channel['name']}_valid"))
        decls.append(create_internal_signal_decl(
            f"{channel['name']}_ready"))

        # Declare extra signals if present
        extra_signals = channel.get("extra_signals", {})
        decls.extend(create_internal_extra_signals_decl(
            channel["name"], extra_signals))
    else:
        # Channel is an array

        # Declare data signal if present
        if channel["bitwidth"] > 0:
            decls.append(create_internal_array_decl(
                channel["name"], channel["bitwidth"], size))

        # Declare handshake signals
        decls.append(create_internal_vector_decl(
            f"{channel['name']}_valid", size))
        decls.append(create_internal_vector_decl(
            f"{channel['name']}_ready", size))

        # Declare extra signals if present
        extra_signals = channel.get("extra_signals", {})
        decls.extend(create_internal_extra_signals_decl(
            channel["name"], extra_signals))

    return decls
