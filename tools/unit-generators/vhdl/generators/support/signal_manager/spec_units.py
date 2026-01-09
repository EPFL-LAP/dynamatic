from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_mapping
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl
from .utils.types import Channel, ExtraSignals


def _generate_concat(channel: Channel, concat_layout: ConcatLayout, concat_assignments: list[str], internal_channel_decls: list[str], internal_channels: dict[str, Channel]):
    # Concat extra signals, but spec is forwarded.

    channel_name = channel["name"]
    internal_name = f"{channel_name}_concat"
    channel_bitwidth = channel["bitwidth"]
    channel_size = channel.get("size", 0)

    internal_channel: Channel = {
        "name": internal_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {"spec": 1}
    }

    # Declare the internal (concatenated) channel
    # Example:
    # signal ins_concat : std_logic_vector(39 downto 0);
    # signal ins_concat_valid : std_logic;
    # signal ins_concat_ready : std_logic;
    # signal ins_concat_spec : std_logic_vector(0 downto 0);
    internal_channel_decls.extend(
        create_internal_channel_decl(internal_channel))

    # Register the internal channel
    internal_channels[channel_name] = internal_channel

    # Concatenate the input channel data and extra signals to create the internal channel
    # Example:
    # ins_concat(32 - 1 downto 0) <= ins;
    # ins_concat(39 downto 32) <= ins_tag0;
    # ins_concat_valid <= ins_valid;
    # ins_ready <= ins_concat_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        channel_name, channel_bitwidth, internal_name, concat_layout, channel_size))

    # Forward spec bit
    # Example: ins_concat_spec <= ins_spec;
    concat_assignments.append(f"{internal_name}_spec <= {channel_name}_spec;")


def _generate_slice(channel: Channel, concat_layout: ConcatLayout, slice_assignments: list[str], internal_channel_decls: list[str], internal_channels: dict[str, Channel]):
    # Slice extra signals, but spec is forwarded.

    channel_name = channel["name"]
    internal_name = f"{channel_name}_concat"
    channel_bitwidth = channel["bitwidth"]
    channel_size = channel.get("size", 0)

    internal_channel: Channel = {
        "name": internal_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {"spec": 1}
    }

    # Declare the internal (concatenated) channel
    # Example:
    # signal outs_concat : std_logic_vector(39 downto 0);
    # signal outs_concat_valid : std_logic;
    # signal outs_concat_ready : std_logic;
    # signal outs_concat_spec : std_logic_vector(0 downto 0);
    internal_channel_decls.extend(
        create_internal_channel_decl(internal_channel))

    # Register the internal channel
    internal_channels[channel_name] = internal_channel

    # Slice the internal channel to create the output channel data and extra signals
    # Example:
    # outs <= outs_concat(32 - 1 downto 0);
    # outs_tag0 <= outs_concat(39 downto 32);
    # outs_valid <= outs_concat_valid;
    # outs_concat_ready <= outs_ready;
    slice_assignments.extend(generate_slice_and_handshake(
        internal_name, channel_name, channel_bitwidth, concat_layout, channel_size))

    # Forward spec bit
    # Example: outs_spec <= outs_concat_spec;
    slice_assignments.append(f"{channel_name}_spec <= {internal_name}_spec;")


def generate_spec_units_signal_manager(
    name: str,
    in_channels: list[Channel],
    out_channels: list[Channel],
    extra_signals_without_spec: ExtraSignals,
    ctrl_names: list[str],
    generate_inner: Callable[[str], str]
):
    """
    Generate a VHDL signal manager for speculative units that handles extra signals
    except for the `spec` signal.

    This function manages the concatenation of extra signals (excluding `spec`) and
    maps them to the inner component while handling tagless control signals
    separately. The control signals are mapped directly to the inner component
    without concatenation.

    Args:
      name: Name for the signal manager entity.
      in_channels: List of input channels for the signal manager.
      out_channels: List of output channels for the signal manager.
      extra_signals_without_spec: List of extra signals (except `spec`) to be handled.
      ctrl_names: List of control signal names that should be separated from data signals.
      generate_inner: Function to generate the inner component.

    Returns:
      A string representing the complete VHDL architecture for the signal manager.
    """

    entity = generate_entity(name, in_channels, out_channels)

    # Separate input channels into control channels and non-control channels
    in_channel_without_ctrl = [
        channel for channel in in_channels if not channel["name"] in ctrl_names]
    ctrl_channels = [
        channel for channel in in_channels if channel["name"] in ctrl_names]

    # Layout info for how extra signals (except for spec) are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals_without_spec)

    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    assignments: list[str] = []
    decls: list[str] = []
    internal_channels: dict[str, Channel] = {}

    for channel in in_channel_without_ctrl:
        _generate_concat(channel, concat_layout,
                         assignments, decls, internal_channels)

    for channel in out_channels:
        _generate_slice(channel, concat_layout,
                        assignments, decls, internal_channels)

    mappings: list[str] = []
    for original_channel_name, channel in internal_channels.items():
        # Internal channel name is different from the original channel name
        mappings.extend(generate_mapping(original_channel_name, channel))

    for ctrl_channel in ctrl_channels:
        # Control channels are not concatenated, just mapped directly
        mappings.extend(generate_mapping(ctrl_channel["name"], ctrl_channel))

    architecture = f"""
-- Architecture of signal manager (spec_units)
architecture arch of {name} is
  {"\n  ".join(decls)}
begin
  -- Concat/slice data and extra signals
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
