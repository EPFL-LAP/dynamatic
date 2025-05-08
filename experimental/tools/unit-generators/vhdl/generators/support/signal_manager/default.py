from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.types import Port, ExtraSignals
from .utils.generation import generate_signal_wise_forwarding, generate_mapping


def _generate_forwarding(in_channel_names: list[str], out_channel_names: list[str], extra_signals: ExtraSignals) -> str:
  extra_signal_assignments = []
  for signal_name, signal_bitwidth in extra_signals.items():
    # Signal-wise forwarding of extra signals from input channels to output channels
    assignments, _ = generate_signal_wise_forwarding(
        in_channel_names, out_channel_names, signal_name, signal_bitwidth)
    extra_signal_assignments.extend(assignments)

  return "\n  ".join(extra_signal_assignments)


def generate_default_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str]
) -> str:
  """
  Generate the full VHDL code for a default signal manager that forwards extra signals from input ports to output ports.

  Args:
    name: Name for the signal manager entity.
    in_ports: List of input ports for the signal manager.
    out_ports: List of output ports for the signal manager.
    extra_signals: Dictionary of extra signals (e.g., spec, tag) to be handled.
    generate_inner: Function to generate the inner component.

  Returns:
    A string representing the complete VHDL architecture for the default signal manager.
  """
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  in_channel_names = [port["name"] for port in in_ports]
  out_channel_names = [port["name"] for port in out_ports]
  extra_signal_assignments = _generate_forwarding(
      in_channel_names, out_channel_names, extra_signals)

  # Map channels to inner component
  mappings = []
  for channel in in_ports + out_ports:
    mappings.extend(generate_mapping(channel, channel["name"]))
  mappings = ",\n      ".join(mappings)

  architecture = f"""
-- Architecture of signal manager (default)
architecture arch of {name} is
begin
  -- Forward extra signals to output ports
  {extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
