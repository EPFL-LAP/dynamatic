from collections.abc import Callable
from .entity import generate_entity
from .concat import ConcatenationInfo, generate_concat_signal_decls, generate_concat_port_assignments
from .mapping import generate_simple_mappings, generate_concat_mappings


def generate_spec_units_signal_manager(name, in_ports, out_ports, extra_signals_without_spec, ctrl_names, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  in_ports_without_ctrl = [
      port for port in in_ports if not port["name"] in ctrl_names]
  ctrl_ports = [
      port for port in in_ports if port["name"] in ctrl_names]
  extra_signal_names_without_spec = [
      signal_name for signal_name in extra_signals_without_spec]

  concat_info = ConcatenationInfo(extra_signals_without_spec)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  concat_signal_decls = generate_concat_signal_decls(
      in_ports_without_ctrl + out_ports, extra_signals_bitwidth)

  concat_logic = generate_concat_port_assignments(
      in_ports_without_ctrl, out_ports, concat_info)

  mappings = generate_concat_mappings(
      in_ports_without_ctrl + out_ports, extra_signals_bitwidth, extra_signal_names_without_spec) + ",\n" + \
      generate_simple_mappings(ctrl_ports)

  architecture = f"""
-- Architecture of signal manager (spec_units)
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
