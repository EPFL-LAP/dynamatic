from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat, generate_slice, generate_mapping, generate_handshake_forwarding
from .utils.types import Port, ExtraSignals


def _generate_concat(channel: Port, concat_layout: ConcatLayout, concat_assignments: list[str], concat_decls: list[str], concat_channels: dict[str, Port]):
  channel_name = channel["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = channel["bitwidth"]
  channel_size = channel.get("size", 0)

  # Concatenate the input channel data and extra signals to create the concat channel
  assignments, decls = generate_concat(
      channel_name, channel_bitwidth, concat_name, concat_layout, channel_size)
  concat_assignments.extend(assignments)
  # Declare the concat channel data signal
  concat_decls.extend(decls["out"])

  # Forward the input channel handshake to the concat channel
  assignments, decls = generate_handshake_forwarding(
      channel_name, concat_name, channel_size)
  concat_assignments.extend(assignments)
  # Declare the concat channel handshake signals
  concat_decls.extend(decls["out"])

  concat_channels[channel_name] = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {"spec": 1}
  }


def _generate_slice(port: Port, concat_layout: ConcatLayout, slice_assignments: list[str], slice_decls: list[str], slice_channels: dict[str, Port]):
  channel_name = port["name"]
  concat_name = f"{channel_name}_concat"
  channel_bitwidth = port["bitwidth"]
  channel_size = port.get("size", 0)

  # Slice the concat channel to create the output channel data and extra signals
  assignments, decls = generate_slice(
      concat_name, channel_name, channel_bitwidth, concat_layout, channel_size)
  slice_assignments.extend(assignments)
  # Declare the concat channel data signal
  slice_decls.extend(decls["in"])

  # Forward the concat channel handshake to the output channel
  assignments, decls = generate_handshake_forwarding(
      concat_name, channel_name, channel_size)
  slice_assignments.extend(assignments)
  # Declare the concat channel handshake signals
  slice_decls.extend(decls["in"])

  slice_channels[channel_name] = {
      "name": concat_name,
      "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
      "size": channel_size,
      "extra_signals": {"spec": 1}
  }


def generate_spec_units_signal_manager(
    name: str,
    in_channels: list[Port],
    out_channels: list[Port],
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
    in_ports: List of input ports for the signal manager.
    out_ports: List of output ports for the signal manager.
    extra_signals_without_spec: List of extra signals (except `spec`) to be handled.
    ctrl_names: List of control signal names that should be separated from data signals.
    generate_inner: Function to generate the inner component.

  Returns:
    A string representing the complete VHDL architecture for the signal manager.
  """

  entity = generate_entity(name, in_channels, out_channels)

  # Separate input ports into control ports and non-control ports
  in_channel_without_ctrl = [
      port for port in in_channels if not port["name"] in ctrl_names]
  ctrl_channels = [
      port for port in in_channels if port["name"] in ctrl_names]

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals_without_spec)

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  assignments = []
  decls = []
  channels = {}

  for port in in_channel_without_ctrl:
    _generate_concat(port, concat_layout,
                     assignments, decls, channels)

  for port in out_channels:
    _generate_slice(port, concat_layout,
                    assignments, decls, channels)

  mappings = []
  for channel_name, channel in channels.items():
    mappings.append(generate_mapping(
        channel, channel_name))

  for ctrl_channel in ctrl_channels:
    mappings.extend(generate_mapping(
        ctrl_channel, ctrl_channel["name"]))

  assignments = "\n  ".join(assignments)
  decls = "\n  ".join(decls)
  mappings = ",\n      ".join(mappings)
  architecture = f"""
-- Architecture of signal manager (spec_units)
architecture arch of {name} is
  {decls}
begin
  -- Concat/slice data and extra signals
  {assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
