def generate_entity(entity_name, in_channels, out_channels) -> str:
    """
    Generate entity for signal manager, based on input and output channels
    """

    # Unify input and output channels, and add direction
    unified_channels = []
    for channel in in_channels:
        unified_channels.append({
            **channel,
            "direction": "in"
        })
    for channel in out_channels:
        unified_channels.append({
            **channel,
            "direction": "out"
        })

    channel_decls = []
    # Add channel declarations for each channel
    for channel in unified_channels:
        dir = channel["direction"]
        ready_dir = "out" if dir == "in" else "in"

        name = channel["name"]
        bitwidth = channel["bitwidth"]
        extra_signals = channel.get("extra_signals", {})
        size = channel.get("size", 0)

        if size == 0:
            # Usual case

            # Generate data signal if present
            if bitwidth > 0:
                channel_decls.append(
                    f"{name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

            channel_decls.append(f"{name}_valid : {dir} std_logic")
            channel_decls.append(f"{name}_ready : {ready_dir} std_logic")

            # Generate extra signals for this input channel
            for signal_name, signal_bitwidth in extra_signals.items():
                channel_decls.append(
                    f"{name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
        else:
            # Generate data_array signal declarations for 2d input channel

            # Generate data signal if present
            if bitwidth > 0:
                channel_decls.append(
                    f"{name} : {dir} data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0)")

            # Use std_logic_vector for valid/ready of 2d input channel
            channel_decls.append(
                f"{name}_valid : {dir} std_logic_vector({size} - 1 downto 0)")
            channel_decls.append(
                f"{name}_ready : {ready_dir} std_logic_vector({size} - 1 downto 0)")

            # Generate extra signal declarations for each item in the 2d input channel
            for i in range(size):
                # The netlist generator declares extra signals independently for each item,
                # in contrast to ready/valid signals.
                for signal_name, signal_bitwidth in extra_signals.items():
                    channel_decls.append(
                        f"{name}_{i}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")

    channel_decls_str = ";\n    ".join(channel_decls).lstrip()

    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity {entity_name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
    {channel_decls_str}
  );
end entity;
"""
