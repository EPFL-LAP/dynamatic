from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.types import Channel, ExtraSignals
from .utils.generation import generate_signal_wise_forwarding, generate_default_mappings, enumerate_channel_names


def _generate_forwarding(in_channel_names: list[str], out_channel_names: list[str], signal_name: str, signal_bitwidth: int, extra_signal_assignments: list[str]):
  # Signal-wise forwarding of extra signals from input channels to output channels
  assignments, _ = generate_signal_wise_forwarding(
      in_channel_names, out_channel_names, signal_name, signal_bitwidth)
  extra_signal_assignments.extend(assignments)


def generate_default_signal_manager(
    name: str,
    in_channels: list[Channel],
    out_channels: list[Channel],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str]
) -> str:
  """
  Generate the full VHDL code for a default signal manager that forwards extra signals from input channels to output channels.

  Args:
    name: Name for the signal manager entity.
    in_channels: List of input channels for the signal manager.
    out_channels: List of output channels for the signal manager.
    extra_signals: Dictionary of extra signals (e.g., spec, tag) to be handled.
    generate_inner: Function to generate the inner component.

  Returns:
    A string representing the complete VHDL architecture for the default signal manager.
  """
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_channels, out_channels)

  in_channel_names = enumerate_channel_names(in_channels)
  out_channel_names = enumerate_channel_names(out_channels)
  extra_signal_assignments = []
  for signal_name, signal_bitwidth in extra_signals.items():
    _generate_forwarding(in_channel_names, out_channel_names,
                         signal_name, signal_bitwidth, extra_signal_assignments)

  # Map channels to inner component
  mappings = generate_default_mappings(in_channels + out_channels)

  architecture = f"""
-- Architecture of signal manager (default)
architecture arch of {name} is
begin
  -- Forward extra signals to output channels
  {"\n  ".join(extra_signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
