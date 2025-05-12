from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_mapping, generate_channel_decls
from .utils.types import Channel, ExtraSignals


def _generate_concat(channel: Channel, concat_layout: ConcatLayout, concat_assignments: list[str], concat_channel_decls: list[str], concat_channels: dict[str, Channel]):
  channel_name = channel["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = channel["bitwidth"]
  channel_size = channel.get("size", 0)

  concat_channel: Channel = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {"spec": 1}
  }

  # Declare the concat channel
  concat_channel_decls.extend(generate_channel_decls(concat_channel))

  # Register the concat channel
  concat_channels[channel_name] = concat_channel

  # Concatenate the input channel data and extra signals to create the concat channel
  concat_assignments.extend(generate_concat_and_handshake(
      channel_name, channel_bitwidth, concat_name, concat_layout, channel_size))

  # Forward spec bit
  concat_assignments.append(f"{concat_name}_spec <= {channel_name}_spec;")


def _generate_slice(channel: Channel, concat_layout: ConcatLayout, slice_assignments: list[str], concat_channel_decls: list[str], concat_channels: dict[str, Channel]):
  channel_name = channel["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = channel["bitwidth"]
  channel_size = channel.get("size", 0)

  concat_channel: Channel = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {"spec": 1}
  }

  # Declare the concat channel
  concat_channel_decls.extend(generate_channel_decls(concat_channel))

  # Register the concat channel
  concat_channels[channel_name] = concat_channel

  # Slice the concat channel to create the output channel data and extra signals
  slice_assignments.extend(generate_slice_and_handshake(
      concat_name, channel_name, channel_bitwidth, concat_layout, channel_size))

  # Forward spec bit
  slice_assignments.append(f"{channel_name}_spec <= {concat_name}_spec;")


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

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals_without_spec)

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  assignments = []
  decls = []
  channels = {}

  for channel in in_channel_without_ctrl:
    _generate_concat(channel, concat_layout,
                     assignments, decls, channels)

  for channel in out_channels:
    _generate_slice(channel, concat_layout,
                    assignments, decls, channels)

  mappings = []
  for channel_name, channel in channels.items():
    mappings.append(generate_mapping(
        channel, channel_name))

  for ctrl_channel in ctrl_channels:
    mappings.extend(generate_mapping(
        ctrl_channel, ctrl_channel["name"]))

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
