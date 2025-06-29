from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_mapping
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl
from .utils.types import Channel, ExtraSignals


def _generate_concat(in_channel: Channel, concat_layout: ConcatLayout, concat_assignments: list[str], internal_channel_decls: list[str], internal_channels: dict[str, Channel]):
    channel_name = in_channel["name"]
    internal_name = f"{channel_name}_concat"
    channel_bitwidth = in_channel["bitwidth"]
    channel_size = in_channel.get("size", 0)

    internal_channel: Channel = {
        "name": internal_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {}
    }

    # Declare the internal (concatenated) channel
    # Example:
    # signal ins_concat : std_logic_vector(32 downto 0);
    # signal ins_concat_valid : std_logic;
    # signal ins_concat_ready : std_logic;
    internal_channel_decls.extend(
        create_internal_channel_decl(internal_channel))

    # Register the internal channel
    internal_channels[channel_name] = internal_channel

    # Concatenate the input channel data and extra signals to create the internal channel
    # Example:
    # ins_concat(32 - 1 downto 0) <= ins;
    # ins_concat(32 downto 32) <= ins_spec;
    # ins_concat_valid <= ins_valid;
    # ins_ready <= ins_concat_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        channel_name,
        channel_bitwidth,
        internal_name,
        concat_layout,
        channel_size
    ))


def _generate_slice(out_channel: Channel, concat_layout: ConcatLayout, slice_assignments: list[str], internal_channel_decls: list[str], internal_channels: dict[str, Channel]):
    channel_name = out_channel["name"]
    internal_name = f"{channel_name}_concat"
    channel_bitwidth = out_channel["bitwidth"]
    channel_size = out_channel.get("size", 0)

    internal_channel: Channel = {
        "name": internal_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {}
    }

    # Declare the internal (concatenated) channel
    # Example:
    # signal outs_concat : data_array(2 downto 0)(32 downto 0);
    # signal outs_concat_valid : std_logic_vector(2 downto 0);
    # signal outs_concat_ready : std_logic_vector(2 downto 0);
    internal_channel_decls.extend(
        create_internal_channel_decl(internal_channel))

    # Register the internal channel
    internal_channels[channel_name] = internal_channel

    # Slice the internal channel to create the output channel data and extra signals
    # Example:
    # outs(0) <= outs_concat(0)(32 - 1 downto 0);
    # outs_0_spec <= outs_concat(0)(32 downto 32);
    # outs(1) <= outs_concat(1)(32 - 1 downto 0);
    # outs_1_spec <= outs_concat(1)(32 downto 32);
    # outs(2) <= outs_concat(2)(32 - 1 downto 0);
    # outs_2_spec <= outs_concat(2)(32 downto 32);
    # outs_valid <= outs_concat_valid;
    # outs_concat_ready <= outs_ready;
    slice_assignments.extend(generate_slice_and_handshake(
        internal_name,
        channel_name,
        channel_bitwidth,
        concat_layout,
        channel_size
    ))


def generate_concat_signal_manager(
    name: str,
    in_channels: list[Channel],
    out_channels: list[Channel],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str]
):
    """
    Generate a signal manager architecture that handles the concatenation of extra signals
    for input and output channels, and forwards them to an inner entity.

    Args:
      name: Name for the signal manager entity.
      in_channels: List of input channels for the signal manager.
      out_channels: List of output channels for the signal manager.
      extra_signals: Dictionary of extra signals (e.g., spec, tag) to be concatenated.
      generate_inner: Function to generate the inner component.

    Returns:
      A string representing the complete VHDL architecture for the signal manager.
    """
    entity = generate_entity(name, in_channels, out_channels)

    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)

    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    assignments: list[str] = []
    decls: list[str] = []
    channels: dict[str, Channel] = {}
    for in_channel in in_channels:
        _generate_concat(in_channel, concat_layout,
                         assignments, decls, channels)

    for out_channel in out_channels:
        _generate_slice(out_channel, concat_layout,
                        assignments, decls, channels)

    mappings: list[str] = []
    for original_channel_name, concat_channel in channels.items():
        # Example:
        # ins => ins_concat,
        # ins_valid => ins_concat_valid,
        # ins_ready => ins_concat_ready,
        mappings.extend(generate_mapping(
            original_channel_name, concat_channel))

    architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  {"\n  ".join(decls)}
begin
  -- Concate/slice data and extra signals
  {"\n  ".join(assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {",\n      ".join(mappings)}
    );
end architecture;
"""

    return inner + entity + architecture
