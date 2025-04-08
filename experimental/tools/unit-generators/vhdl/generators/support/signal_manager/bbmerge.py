from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.forwarding import get_default_extra_signal_value
from .utils.concat import generate_concat_signal_decls, ConcatenationInfo, generate_concat_port_assignments
from .utils.mapping import generate_inner_port_mapping, generate_concat_mappings


def _generate_bbmerge_lacking_spec_statements(spec_inputs, size, data_in_name):
  """
  e.g.,
  - decls: signal lhs_0_spec : std_logic_vector(0 downto 0);
  - assigns: lhs_0_spec <= "0";
  """
  # Declare and assign default spec bits for inputs without them
  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"  signal {data_in_name}_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  lacking_spec_port_assignments = [
      f"  {data_in_name}_{i}_spec <= {get_default_extra_signal_value("spec")};" for i in lacking_spec_ports
  ]
  return "\n".join(lacking_spec_port_decls).lstrip(), "\n".join(lacking_spec_port_assignments).lstrip()


def generate_mux_signal_manager(name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, index_extra_signals, spec_inputs, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  in_ports_without_index = [
      port for port in in_ports if port["name"] != index_name]
  index_port = [
      port for port in in_ports if port["name"] == index_name][0]

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(out_extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  lacking_spec_port_decls, lacking_spec_port_assignments = _generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, data_in_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      in_ports_without_index + out_ports, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments(
      in_ports_without_index, out_ports, concat_info)

  # Port forwarding for the inner entity
  mappings = generate_concat_mappings(
      in_ports_without_index + out_ports, extra_signals_bitwidth) + ",\n" + \
      ",\n".join(generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (mux)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

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


def _generate_cmerge_index_extra_signal_assignments(index_name, index_extra_signals) -> str:
  """
  e.g., index_tag0 <= "0";
  """
  # TODO: Extra signals for index port are not tested
  if index_extra_signals:
    index_extra_signals_list = []
    for signal_name in index_extra_signals:
      index_extra_signals_list.append(
          f"  {index_name}_{signal_name} <= {get_default_extra_signal_value(signal_name)};")
    return "\n".join(index_extra_signals_list)
  return ""


def generate_cmerge_signal_manager(name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, index_extra_signals, spec_inputs, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  out_ports_without_index = [
      port for port in out_ports if port["name"] != index_name]
  index_port = [
      port for port in out_ports if port["name"] == index_name][0]

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(out_extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  lacking_spec_port_decls, lacking_spec_port_assignments = _generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, data_in_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports_without_index, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments(
      in_ports, out_ports_without_index, concat_info)

  # Assign index extra signals
  index_extra_signal_assignments = _generate_cmerge_index_extra_signal_assignments(
      index_name, index_extra_signals)

  # Port forwarding for the inner entity
  mappings = generate_concat_mappings(
      in_ports + out_ports_without_index, extra_signals_bitwidth) + ",\n" + \
      ",\n".join(generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (cmerge)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

  -- Concatenate data and extra signals
  {concat_logic}

  -- Assign index extra signals (if any)
  {index_extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
