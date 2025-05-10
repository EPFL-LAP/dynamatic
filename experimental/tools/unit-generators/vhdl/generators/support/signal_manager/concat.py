from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat, generate_slice, generate_mapping, generate_handshake_forwarding
from .utils.types import Port, ExtraSignals


def _generate_concat(in_channel: Port, concat_layout: ConcatLayout, concat_assignments: list[str], concat_decls: list[str], concat_channels: dict[str, Port]):
  channel_name = in_channel["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = in_channel["bitwidth"]
  channel_size = in_channel.get("size", 0)

  # Concatenate the input channel data and extra signals to create the concat channel
  assignments, declarations = generate_concat(
      channel_name,
      channel_bitwidth,
      concat_name,
      concat_layout,
      channel_size
  )
  concat_assignments.extend(assignments)
  # Declare the concat channel data signal
  concat_decls.extend(declarations[concat_name])

  # Forward the input channel handshake to the concat channel
  assignments, declarations = generate_handshake_forwarding(
      channel_name, concat_name, channel_size)
  concat_assignments.extend(assignments)
  # Declare the concat channel handshake signals
  concat_decls.extend(declarations[concat_name])

  concat_channels[channel_name] = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {}
  }


def _generate_slice(out_channel: Port, concat_layout: ConcatLayout, slice_assignments: list[str], slice_decls: list[str], slice_channels: dict[str, Port]):
  channel_name = out_channel["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = out_channel["bitwidth"]
  channel_size = out_channel.get("size", 0)

  # Slice the concat channel to create the output channel data and extra signals
  assignments, declarations = generate_slice(
      concat_name,
      channel_name,
      channel_bitwidth,
      concat_layout,
      channel_size
  )
  slice_assignments.extend(assignments)
  # Declare the concat channel data signal
  slice_decls.extend(declarations[concat_name])

  # Forward the concat channel handshake to the output channel
  assignments, declarations = generate_handshake_forwarding(
      concat_name, channel_name, channel_size)
  slice_assignments.extend(assignments)
  # Declare the concat channel handshake signals
  slice_decls.extend(declarations[concat_name])

  slice_channels[channel_name] = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {}
  }


def generate_concat_signal_manager(
    name: str,
    in_channels: list[Port],
    out_channels: list[Port],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str]
):
  """
  Generate a signal manager architecture that handles the concatenation of extra signals
  for input and output ports, and forwards them to an inner entity.

  Args:
    name: Name for the signal manager entity.
    in_ports: List of input ports for the signal manager.
    out_ports: List of output ports for the signal manager.
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

  assignments = []
  decls = []
  channels = {}
  for in_channel in in_channels:
    _generate_concat(in_channel, concat_layout,
                     assignments, decls, channels)

  for out_channel in out_channels:
    _generate_slice(out_channel, concat_layout,
                    assignments, decls, channels)

  mappings = []
  for channel_name, concat_channel in channels.items():
    mappings.extend(generate_mapping(concat_channel, channel_name))

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
