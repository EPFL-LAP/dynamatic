from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.forwarding import generate_forwarding_assignments
from .utils.types import Port, ExtraSignals
from .utils.mapping import generate_simple_mappings, get_unhandled_extra_signals


def _generate_normal_signal_assignments(in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals) -> str:
  extra_signal_assignments = generate_forwarding_assignments(
      [port["name"] for port in in_ports],
      [port["name"] for port in out_ports],
      list(extra_signals)
  )

  return "\n".join(extra_signal_assignments).lstrip()


def generate_normal_signal_manager(name: str, in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals, generate_inner: Callable[[str], str]) -> str:
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  extra_signal_assignments = _generate_normal_signal_assignments(
      in_ports, out_ports, extra_signals)

  unhandled_extra_signals = get_unhandled_extra_signals(
      in_ports + out_ports, extra_signals)
  mappings = generate_simple_mappings(
      in_ports + out_ports, unhandled_extra_signals)

  architecture = f"""
-- Architecture of signal manager (normal)
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
