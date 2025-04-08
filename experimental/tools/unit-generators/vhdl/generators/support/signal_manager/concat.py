from collections.abc import Callable
from .utils.mapping import generate_concat_mappings, get_unhandled_extra_signals
from .utils.entity import generate_entity
from .utils.concat import ConcatenationInfo, generate_concat_signal_decls_from_ports, generate_concat_port_assignments_from_ports
from .utils.types import Port, ExtraSignals


def generate_concat_signal_manager(name: str, in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls_from_ports(
      in_ports + out_ports, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments_from_ports(
      in_ports, out_ports, concat_info)

  # Port forwarding for the inner entity
  unhandled_extra_signals = get_unhandled_extra_signals(
      in_ports + out_ports, extra_signals)
  mappings = generate_concat_mappings(
      in_ports + out_ports, extra_signals_bitwidth, unhandled_extra_signals)

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
