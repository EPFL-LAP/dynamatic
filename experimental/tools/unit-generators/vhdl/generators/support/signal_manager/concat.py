from collections.abc import Callable
from .utils.mapping import generate_concat_mappings, get_unhandled_extra_signals
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout, generate_concat_signal_decls_from_ports, generate_concat_port_assignments_from_ports
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

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = "\n  ".join(generate_concat_signal_decls_from_ports(
      in_ports + out_ports, extra_signals_bitwidth))

  # Assign inner concatenated signals
  concat_logic = "\n  ".join(generate_concat_port_assignments_from_ports(
      in_ports, out_ports, concat_layout))

  # Map concatenated ports and untouched extra signals to inner component
  unhandled_extra_signals = get_unhandled_extra_signals(
      in_ports + out_ports, extra_signals)
  mappings = "\n      ".join(generate_concat_mappings(
      in_ports + out_ports, extra_signals_bitwidth, unhandled_extra_signals))

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
