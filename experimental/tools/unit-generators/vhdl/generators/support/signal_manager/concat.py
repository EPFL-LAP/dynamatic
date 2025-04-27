from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout, generate_concat, generate_slice, subtract_extra_signals, generate_signal_direct_forwarding, generate_mapping, generate_handshake_forwarding
from .utils.types import Port, ExtraSignals


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
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  concat_ports = {}
  concat_signal_decls = []
  concat_logic = []
  for in_port in in_ports:
    channel_name = in_port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = in_port["bitwidth"]
    channel_size = in_port.get("size", 0)
    channel_extra_signals = in_port.get("extra_signals", {})

    assignments, declarations = generate_concat(
        channel_name,
        channel_bitwidth,
        concat_name,
        concat_layout,
        channel_size
    )
    concat_signal_decls.extend(declarations["out"])
    concat_logic.extend(assignments)

    assignments, declarations = generate_handshake_forwarding(
        channel_name, concat_name, channel_size)
    concat_signal_decls.extend(declarations["out"])
    concat_logic.extend(assignments)

    unhandled_extra_signals = subtract_extra_signals(
        channel_extra_signals, extra_signals)
    for signal_name, signal_bitwidth in unhandled_extra_signals.items():
      assignments, declarations = generate_signal_direct_forwarding(
          channel_name, concat_name, signal_name, signal_bitwidth)
      concat_signal_decls.extend(declarations["out"])
      concat_logic.extend(assignments)

    concat_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": unhandled_extra_signals
    }

  for out_port in out_ports:
    channel_name = out_port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = out_port["bitwidth"]
    channel_size = out_port.get("size", 0)
    channel_extra_signals = out_port.get("extra_signals", {})

    assignments, declarations = generate_slice(
        concat_name,
        channel_name,
        channel_bitwidth,
        concat_layout,
        channel_size
    )
    concat_signal_decls.extend(declarations["in"])
    concat_logic.extend(assignments)

    assignments, declarations = generate_handshake_forwarding(
        concat_name, channel_name, channel_size)
    concat_signal_decls.extend(declarations["in"])
    concat_logic.extend(assignments)

    unhandled_extra_signals = subtract_extra_signals(
        channel_extra_signals, extra_signals)
    for signal_name, signal_bitwidth in unhandled_extra_signals.items():
      assignments, declarations = generate_signal_direct_forwarding(
          concat_name, channel_name, signal_name, signal_bitwidth)
      concat_signal_decls.extend(declarations["out"])
      concat_logic.extend(assignments)

    concat_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": unhandled_extra_signals
    }

  concat_signal_decls = "\n  ".join(concat_signal_decls)
  concat_logic = "\n  ".join(concat_logic)

  mappings = []
  for original_name, concat_channel in concat_ports.items():
    mappings.extend(generate_mapping(concat_channel, original_name))
  mappings = ",\n      ".join(mappings)

  architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
