def generate_entity(entity_name, in_channels, out_channels) -> str:
    """
    Generate entity for signal manager, based on input and output channels
    """

    # Unify input and output channels, and add direction
    unified_channels = []
    for channel in in_channels:
        unified_channels.append({
            **channel,
            "direction": "input"
        })
    for channel in out_channels:
        unified_channels.append({
            **channel,
            "direction": "output"
        })

    channel_decls = []
    # Add channel declarations for each channel
    for channel in unified_channels:
        dir = channel["direction"]
        ready_dir = "output" if dir == "input" else "input"

        name = channel["name"]
        bitwidth = channel["bitwidth"]
        extra_signals = channel.get("extra_signals", {})
        size = channel.get("size", 0)

        if size == 0:
            # Usual case

            # Generate data signal if present
            if bitwidth > 0:
                channel_decls.append(
                    f"{dir} [{bitwidth} - 1:0] {name}")

            channel_decls.append(f"{dir} {name}_valid")
            channel_decls.append(f"{ready_dir} {name}_ready")

            # Generate extra signals for this input channel
            for signal_name, signal_bitwidth in extra_signals.items():
                channel_decls.append(
                    f"{dir} [{signal_bitwidth} - 1:0] {name}_{signal_name}")
        else:
            # Generate data_array signal declarations for 2d input channel

            # Generate data signal if present
            if bitwidth > 0:
                channel_decls.append(
                    f"{dir} wire [{bitwidth} - 1:0] {name} [{size} - 1 : 0];")

            # Use std_logic_vector for valid/ready of 2d input channel
            channel_decls.append(
                f"{dir} wire [{size} - 1 : 0] {name}_valid;")
            channel_decls.append(
                f"{ready_dir} wire [{size} - 1 : 0] {name}_ready;")

            # Generate extra signal declarations for each item in the 2d input channel
            for i in range(size):
                # The netlist generator declares extra signals independently for each item,
                # in contrast to ready/valid signals.
                for signal_name, signal_bitwidth in extra_signals.items():
                    channel_decls.append(
                        f"{dir} wire [{signal_bitwidth} - 1 : 0] {name}_{i}_{signal_name};")

    channel_decls_str = ",\n    ".join(channel_decls).lstrip()
    return f"""
// Module of signal manager
module {entity_name} (
    input clk,
    input rst,
    {channel_decls_str}
);
"""
