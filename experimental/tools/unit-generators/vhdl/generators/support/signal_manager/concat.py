from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout
from .utils.generation import generate_concat, generate_slice, generate_mapping, generate_handshake_forwarding
from .utils.types import Port, ExtraSignals


def _generate_concat(in_ports: list[Port], concat_layout: ConcatLayout) -> tuple[str, str, dict[str, Port]]:
  concat_assignments = []
  concat_decls = []
  concat_ports = {}

  for in_port in in_ports:
    channel_name = in_port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = in_port["bitwidth"]
    channel_size = in_port.get("size", 0)

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
    concat_decls.extend(declarations["out"])

    # Forward the input channel handshake to the concat channel
    assignments, declarations = generate_handshake_forwarding(
        channel_name, concat_name, channel_size)
    concat_assignments.extend(assignments)
    # Declare the concat channel handshake signals
    concat_decls.extend(declarations["out"])

    concat_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {}
    }

  return "\n  ".join(concat_assignments), "\n  ".join(concat_decls), concat_ports


def _generate_slice(out_ports: list[Port], concat_layout: ConcatLayout) -> tuple[str, str, dict[str, Port]]:
  slice_assignments = []
  slice_decls = []
  slice_ports = {}

  for out_port in out_ports:
    channel_name = out_port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = out_port["bitwidth"]
    channel_size = out_port.get("size", 0)

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
    slice_decls.extend(declarations["in"])

    # Forward the concat channel handshake to the output channel
    assignments, declarations = generate_handshake_forwarding(
        concat_name, channel_name, channel_size)
    slice_assignments.extend(assignments)
    # Declare the concat channel handshake signals
    slice_decls.extend(declarations["in"])

    slice_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {}
    }

  return "\n  ".join(slice_assignments), "\n  ".join(slice_decls), slice_ports


def _generate_mappings(concat_ports: dict[str, Port], slice_ports: dict[str, Port]) -> str:
  mappings = []
  for original_name, concat_channel in (concat_ports | slice_ports).items():
    mappings.extend(generate_mapping(concat_channel, original_name))
  return ",\n      ".join(mappings)


def generate_concat_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
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
  entity = generate_entity(name, in_ports, out_ports)

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals)

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  concat_assignments, concat_decls, concat_ports = _generate_concat(
      in_ports, concat_layout)
  slice_assignments, slice_decls, slice_ports = _generate_slice(
      out_ports, concat_layout)

  mappings = _generate_mappings(concat_ports, slice_ports)

  architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  {concat_decls}
  {slice_decls}
begin
  -- Concate/slice data and extra signals
  {concat_assignments}
  {slice_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
